#!/usr/bin/env python3
# train.py

import os
import sys
import random
import joblib
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from config import Config
from data_loader import YiduS4KDataLoader
from kg_builder import MedicalKGBuilder
from entity_alignment import MedicalEntityAligner

from rapidfuzz import fuzz


# -----------------------------
# DB utilities
# -----------------------------

def load_entities_from_db(db_path: str) -> List[Dict]:
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, name, entity_type FROM entities")
        rows = cur.fetchall()
        return [{'id': r['id'], 'name': r['name'], 'type': r['entity_type']} for r in rows]
    finally:
        conn.close()


def ensure_entities_in_db(yidu_dir: str, db_path: str, limit_entities: int = None) -> List[Dict]:
    """
    若 DB 为空，则从 Yidu-S4K 初始化实体
    """
    entities = load_entities_from_db(db_path)
    if entities:
        return entities

    print("[WARN] 数据库中未发现实体，尝试从 Yidu-S4K 提取并写入数据库...")

    loader = YiduS4KDataLoader(yidu_dir)
    extracted = loader.extract_entities_from_task1(limit=limit_entities)

    if not extracted:
        print("[ERROR] 无法从 Yidu-S4K 提取实体")
        return []

    builder = MedicalKGBuilder(db_path)
    batch = []
    for e in extracted:
        batch.append({
            'name': e['name'],
            'type': e['type'],
            'aliases': e.get('aliases', [])
        })
    builder.add_entities_batch(batch)

    return load_entities_from_db(db_path)


# -----------------------------
# Training data generation
# -----------------------------


# ======================
# 规则归一化（可继续加）
# ======================
def normalize_medical(text: str) -> str:
    if not text:
        return ""
    rules = [
        "根治性", "辅助", "腹腔镜", "手术", "治疗",
        "切除", "术后", "术", "伴", "并"
    ]
    for r in rules:
        text = text.replace(r, "")
    return text.strip()


def generate_training_pairs_v2(
    yidu_dir: str,
    db_path: str,
    limit_entities: int = None,
    neg_ratio: int = 3,
    exact_pos_ratio: float = 0.2
) -> List[Tuple]:
    """
    生成训练样本对（Ranking-friendly）
    返回：(mention_text, mention_type, entity_name, entity_type, label)
    """

    loader = YiduS4KDataLoader(yidu_dir)

    # -------- 1. 提取 mentions --------
    mentions = []
    for rec in loader.load_task1_training(split='all'):
        text = rec.get('originalText') or rec.get('text') or ''
        for ent in rec.get('entities', []):
            m_text = ent.get('text') or ent.get('entity_text')
            m_type = loader.ENTITY_TYPES.get(
                ent.get('type') or ent.get('entity_label'), 'unknown'
            )
            if m_text:
                mentions.append({
                    "text": m_text,
                    "type": m_type,
                    "context": text
                })

    if not mentions:
        print("[ERROR] 未抽取到 mention")
        return []

    # -------- 2. 加载实体（含 aliases）--------
    entities = ensure_entities_in_db(yidu_dir, db_path, limit_entities)
    if not entities:
        print("[ERROR] 实体库为空")
        return []

    # name / alias → entity 映射
    name2entities = {}
    alias2entities = {}

    for e in entities:
        name2entities.setdefault(e['name'], []).append(e)
        for a in e.get('aliases', []):
            alias2entities.setdefault(a, []).append(e)

    pairs = []
    exact_pos_cnt = 0

    # -------- 3. 构造样本 --------
    for m in tqdm(mentions, desc="生成样本(v2)"):
        m_text = m['text']
        m_type = m['type']

        pos_entities = []

        # (1) alias 正样本（最重要）
        if m_text in alias2entities:
            pos_entities.extend(alias2entities[m_text])

        # (2) exact match（限制比例）
        if m_text in name2entities:
            if random.random() < exact_pos_ratio:
                pos_entities.extend(name2entities[m_text])
                exact_pos_cnt += 1

        # (3) 规则弱同义正样本
        norm_m = normalize_medical(m_text)
        for e in entities:
            if norm_m and norm_m == normalize_medical(e['name']):
                pos_entities.append(e)

        # 去重
        pos_entities = {
            e['id']: e for e in pos_entities
        }.values()

        # 写入正样本
        for pe in pos_entities:
            pairs.append((
                m_text, m_type,
                pe['name'], pe['type'],
                1
            ))

        # -------- 4. Hard Negative --------
        neg_pool = [e for e in entities if e['name'] != m_text]

        hard_negs = []
        for e in neg_pool:
            s = fuzz.token_set_ratio(m_text, e['name'])
            if 50 <= s < 90:
                hard_negs.append(e)

        if len(hard_negs) < neg_ratio:
            hard_negs = random.sample(
                neg_pool,
                min(len(neg_pool), neg_ratio * 2)
            )

        neg_samples = random.sample(
            hard_negs,
            min(len(hard_negs), neg_ratio)
        )

        for ne in neg_samples:
            pairs.append((
                m_text, m_type,
                ne['name'], ne['type'],
                0
            ))

    print(f"[INFO] v2 样本数={len(pairs)} | exact_pos_used={exact_pos_cnt}")
    return pairs


# -----------------------------
# Feature extraction
# -----------------------------

def extract_features_for_pairs(pairs, aligner: MedicalEntityAligner):
    X, y = [], []
    for a, at, b, bt, label in tqdm(pairs, desc="提取特征"):
        feat = aligner.extract_features(
            {'name': a, 'type': at},
            {'name': b, 'type': bt}
        )
        X.append(feat)
        y.append(label)
    return np.vstack(X), np.array(y)


# -----------------------------
# Leak-safe training
# -----------------------------

def train_and_save_model(pairs, aligner, model_path):
    import random
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    import joblib

    # ===== 1. mention 级切分 =====
    mentions = list(set(p[0] for p in pairs))
    random.shuffle(mentions)

    split = int(0.8 * len(mentions))
    train_m = set(mentions[:split])
    test_m  = set(mentions[split:])

    train_pairs = [p for p in pairs if p[0] in train_m]
    test_pairs  = [p for p in pairs if p[0] in test_m]

    # ===== 2. 分别提特征（关键！）=====
    X_tr, y_tr = extract_features_for_pairs(train_pairs, aligner)
    X_te, y_te = extract_features_for_pairs(test_pairs, aligner)

    print(f"[DEBUG] train_pairs={len(train_pairs)} test_pairs={len(test_pairs)}")

    # ===== 3. 训练 =====
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        n_jobs=-1,
        random_state=42
    )

    print("[INFO] 开始训练模型（mention 级切分）...")
    clf.fit(X_tr, y_tr)

    # ===== 4. 评估 =====
    preds = clf.predict(X_te)
    p, r, f, _ = precision_recall_fscore_support(
        y_te, preds, average='binary', zero_division=0
    )
    acc = accuracy_score(y_te, preds)

    print(f"[EVAL] acc={acc:.4f} p={p:.4f} r={r:.4f} f={f:.4f}")

    joblib.dump(clf, model_path)
    print(f"[INFO] 模型已保存到 {model_path}")

# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":

    Config.ensure_directories()

    yidu_dir = os.getenv("YIDU_S4K_PATH", str(Config.YIDU_S4K_DIR))
    db_path = str(Config.DB_PATH)
    model_path = os.path.join(str(Config.MODEL_DIR), "aligner_rf.joblib")
    Path(Config.MODEL_DIR).mkdir(exist_ok=True)

    pairs = generate_training_pairs_v2(yidu_dir, db_path, limit_entities=2000, neg_ratio=3)
    if not pairs:
        sys.exit(1)

    aligner = MedicalEntityAligner()
    X, y = extract_features_for_pairs(pairs, aligner)

    train_and_save_model(pairs, aligner, model_path)
