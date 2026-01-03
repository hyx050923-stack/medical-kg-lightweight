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
        # [修改] 使用 LEFT JOIN 关联查询别名表
        cur.execute("""
            SELECT e.id, e.name, e.entity_type, GROUP_CONCAT(a.alias, '|') as aliases
            FROM entities e
            LEFT JOIN aliases a ON e.id = a.entity_id
            GROUP BY e.id
        """)
        rows = cur.fetchall()
        
        results = []
        for r in rows:
            # 处理 GROUP_CONCAT 可能产生的 None
            alias_list = r['aliases'].split('|') if r['aliases'] else []
            results.append({
                'id': r['id'], 
                'name': r['name'], 
                'type': r['entity_type'],
                'aliases': alias_list
            })
        return results
    except sqlite3.OperationalError:
        return []
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
    from entity_alignment import MedicalEntityAligner
    abbr_map = MedicalEntityAligner.ABBREVIATION_MAP
    
    print(f"[INFO] 正在生成缩写增强样本 (Map size: {len(abbr_map)})...")
    
    for abbr, full_name in abbr_map.items():
        # 在库里找全称对应的实体
        targets = name2entities.get(full_name)
        if targets:
            # 取第一个匹配的实体作为目标
            target = targets[0] 
            
            # 【正样本】: (HTN, disease, 高血压, disease, 1)
            pairs.append((
                abbr, target['type'],
                target['name'], target['type'],
                1
            ))
            
            # 【负样本】: 找一个同类型但名字不对的实体 (增强抗干扰能力)
            # 例如: (HTN, disease, 糖尿病, disease, 0)
            neg_candidates = [e for e in entities if e['type'] == target['type'] and e['name'] != full_name]
            if neg_candidates:
                # 随机抽 2 个负例
                for neg in random.sample(neg_candidates, min(len(neg_candidates), 2)):
                    pairs.append((
                        abbr, target['type'],
                        neg['name'], neg['type'],
                        0
                    ))
    
    print(f"[INFO] 增强后样本数={len(pairs)}")
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

    # 返回测试集对，供后续 Ranking 评估使用
    return clf, test_pairs

def evaluate_ranking_metrics(test_pairs, clf, aligner, top_ks=[1, 3, 5]):
    """
    针对 Ranking 逻辑的评估函数
    """
    from collections import defaultdict
    
    # 1. 按 mention 分组：将属于同一个 mention 的所有候选实体聚在一起
    mention_groups = defaultdict(list)
    for p in test_pairs:
        # p = (m_text, m_type, e_name, e_type, label)
        mention_groups[(p[0], p[1])].append(p)

    hits = {k: 0 for k in top_ks}
    mrr_sum = 0
    total_valid_mentions = 0

    for (m_text, m_type), group in tqdm(mention_groups.items(), desc="Ranking 进度"): # 加入 tqdm 进度条
        if not any(item[4] == 1 for item in group):
            continue
        
        total_valid_mentions += 1
        
        # --- 优化点：批量提取特征并预测 ---
        # 一次性提取该 Mention 下所有候选的特征
        X_group = []
        for m_t, m_tp, e_n, e_tp, label in group:
            feat = aligner.extract_features({'name': m_t, 'type': m_tp}, {'name': e_n, 'type': e_tp})
            X_group.append(feat)
        
        # 2.批量预测该组的所有概率
        probs = clf.predict_proba(np.array(X_group))[:, 1]
        
        scored_candidates = []
        for i, (m_t, m_tp, e_n, e_tp, label) in enumerate(group):
            scored_candidates.append((probs[i], label))
        
        # 3. 按概率从大到小排序
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # 4. 找到第一个正样本的排名
        rank = -1
        for i, (prob, label) in enumerate(scored_candidates):
            if label == 1:
                rank = i + 1
                break
        
        # 5. 累加指标
        if rank != -1:
            for k in top_ks:
                if rank <= k:
                    hits[k] += 1
            mrr_sum += 1.0 / rank

    # 打印结果
    print("\n" + "="*40)
    print(f"Ranking 评估结果 (样本数: {total_valid_mentions})")
    for k in top_ks:
        print(f"Hits@{k}: {hits[k]/total_valid_mentions:.4f}")
    print(f"MRR: {mrr_sum/total_valid_mentions:.4f}")
    print("="*40)

# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":

    Config.ensure_directories()

    yidu_dir = os.getenv("YIDU_S4K_PATH", str(Config.YIDU_S4K_DIR))
    db_path = str(Config.DB_PATH)
    model_path = os.path.join(str(Config.MODEL_DIR), "aligner_rf.joblib")
    Path(Config.MODEL_DIR).mkdir(exist_ok=True)

    pairs = generate_training_pairs_v2(yidu_dir, db_path, limit_entities=None, neg_ratio=3)
    if not pairs:
        sys.exit(1)

    aligner = MedicalEntityAligner()
    X, y = extract_features_for_pairs(pairs, aligner)

    # 训练并获取测试集
    clf, test_pairs = train_and_save_model(pairs, aligner, model_path)
    
    # 执行 Ranking 评估
    print("\n[INFO] 开始进行 Ranking 维度评估...")
    evaluate_ranking_metrics(test_pairs, clf, aligner)

