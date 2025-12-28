# train_alignment.py
import os
import joblib
import random
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from data_loader import YiduS4KDataLoader
from kg_builder import MedicalKGBuilder
from entity_alignment import MedicalEntityAligner
from config import Config

def build_entity_list_from_db(db_path):
    from sqlite3 import connect
    conn = connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, name, entity_type FROM entities")
    rows = cur.fetchall()
    conn.close()
    return [{'id': r[0], 'name': r[1], 'type': r[2]} for r in rows]

def generate_training_pairs(yidu_dir, db_path, limit_entities=None, neg_ratio=3):
    loader = YiduS4KDataLoader(yidu_dir)
    # extract mentions from Yidu training set (文本中的标注)
    mentions = []
    for rec in loader.load_task1_training(split='all'):
        text = rec.get('originalText') or rec.get('text') or ''
        for ent in rec.get('entities', []):
            mention_text = ent.get('text') or ent.get('entity_text') or ''
            mention_type = loader.ENTITY_TYPES.get(ent.get('type'), 'unknown')
            mentions.append({'text': mention_text, 'type': mention_type, 'context': text})
    # load entities from DB
    entities = build_entity_list_from_db(db_path)
    if limit_entities:
        entities = entities[:limit_entities]

    # map name->entity for quick exact-match positives
    name2entities = {}
    for e in entities:
        name2entities.setdefault(e['name'], []).append(e)

    pairs = []
    for m in tqdm(mentions, desc='生成样本'):
        # positive if exact name match exists
        pos_candidates = name2entities.get(m['text'], [])
        for pe in pos_candidates:
            pairs.append((m['text'], m['type'], pe['name'], pe['type'], 1))
        # generate negatives by sampling random entities
        for _ in range(neg_ratio):
            ne = random.choice(entities)
            # avoid accidental positive
            if ne['name'] == m['text']:
                continue
            pairs.append((m['text'], m['type'], ne['name'], ne['type'], 0))
    return pairs

def extract_features_for_pairs(pairs, aligner: MedicalEntityAligner):
    X = []
    y = []
    for a, a_type, b, b_type, label in tqdm(pairs, desc='提取特征'):
        e1 = {'name': a, 'type': a_type}
        e2 = {'name': b, 'type': b_type}
        feat = aligner.extract_features(e1, e2)
        X.append(feat)
        y.append(label)
    return np.vstack(X), np.array(y)

def train_and_save_model(X, y, model_path):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:,1]
    p, r, f, _ = precision_recall_fscore_support(y_test, preds, average='binary')
    acc = accuracy_score(y_test, preds)
    print(f"Eval - acc:{acc:.4f} p:{p:.4f} r:{r:.4f} f:{f:.4f}")
    joblib.dump(clf, model_path)
    print(f"模型已保存到 {model_path}")
    return clf

if __name__ == '__main__':
    Config.ensure_directories()
    yidu_dir = os.getenv('YIDU_S4K_PATH', Config.YIDU_S4K_DIR)
    db_path = str(Config.DB_PATH)
    pairs = generate_training_pairs(yidu_dir, db_path, limit_entities=2000, neg_ratio=3)
    aligner = MedicalEntityAligner()
    X, y = extract_features_for_pairs(pairs, aligner)
    model = train_and_save_model(X, y, os.path.join(str(Config.MODEL_DIR), 'aligner_rf.joblib'))