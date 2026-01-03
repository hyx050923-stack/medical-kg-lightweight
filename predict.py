#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import joblib
from typing import List, Dict

from config import Config
from entity_alignment import MedicalEntityAligner
from entity_linker import UniversalEntityLinker
from entity_recognizer import MedicalEntityRecognizer
from logger_config import setup_logging

STOP_MENTIONS = {"切除", "检查", "手术", "治疗", "病变", "炎症", "组织", "症状", "病史"}

# ------------------------------------------------
# Entity Ranking 核心函数
# ------------------------------------------------
def rank_entities_for_mention(
    mention_text: str,
    mention_type: str,
    candidates: List[Dict],
    clf,
    aligner,
    topk: int = 5
) -> List[Dict]:
    """
    对一个 mention 的候选实体进行打分排序
    """
    scored = []

    for e in candidates:
        feat = aligner.extract_features(
            {"name": mention_text, "type": mention_type},
            {"name": e["name"], "type": e["type"]}
        )

        prob = clf.predict_proba([feat])[0][1]

        scored.append({
            "entity_id": e["id"],
            "entity_name": e["name"],
            "entity_type": e["type"],
            "score": float(prob)
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:topk]


# ------------------------------------------------
# 主预测函数
# ------------------------------------------------
# predict.py

def predict_text(
    text: str,
    clf,
    aligner,
    linker,
    topk: int = 3,
    min_score: float = 0.4  
) -> List[Dict]:
    """
    端到端预测：输入文本 -> 识别 -> 链接
    """
    # 1. 识别实体
    recognizer = MedicalEntityRecognizer()
    mentions = recognizer.recognize(text)
    
    results = []
    for mention_text, mention_type, span in mentions:
        # 【新增】停用词过滤：解决 "切除" -> "xx切除术" 的乱连问题
        if mention_text in STOP_MENTIONS:
            results.append({
                "mention": mention_text,
                "span": span,
                "linked_entity": None,
                "candidates": []
            })
            continue
        # 2. 获取候选
        candidates = linker.get_candidates(
            mention_text, 
            mention_type,
            max_candidates=15
        )
        
        if not candidates:
            results.append({
                "mention": mention_text,
                "span": span,
                "linked_entity": None,
                "candidates": []
            })
            continue

        # 3. 排序
        ranked = rank_entities_for_mention(
            mention_text,
            mention_type,
            candidates,
            clf,
            aligner,
            topk=topk
        )
        
        # =======================================================
        # 解决 "心肌梗死" 匹配 "急性心肌梗死" 分数过低的问题
        # =======================================================
        norm_mention = aligner._normalize_abbreviation(mention_text.upper())

        for r in ranked:
            
            # --- 策略 1: 归一化/缩写救援 (最高优先级) ---
            # 如果归一化后名字完全一样 (HTN -> 高血压 == 高血压)
            # 这代表命中了我们定义的规则字典，直接置信度拉满
            if norm_mention == r['entity_name']:
                r['score'] = 1.0  # 直接满分
                
            # --- 策略 2: 包含关系救援 (次优先级) ---
            # 解决 "心肌梗死" 匹配 "急性心肌梗死" (长词包短词)
            elif len(mention_text) >= 2 and mention_text in r['entity_name']:
                r['score'] += 0.5  # 强行加分
                if r['score'] > 1.0: r['score'] = 1.0

        # 重新排序（因为分数变了）
        ranked.sort(key=lambda x: x["score"], reverse=True)
        # =======================================================

        best = ranked[0]

        # 4. 决策
        if best["score"] >= min_score:
            linked = best
        else:
            linked = None

        results.append({
            "mention": mention_text,
            "span": span,
            "linked_entity": linked,
            "candidates": ranked
        })

    return results

# ------------------------------------------------
# CLI 入口
# ------------------------------------------------
if __name__ == "__main__":
    setup_logging()
    
    # 1. 确保加载的是你刚刚训练好的新模型
    model_path = os.path.join(str(Config.MODEL_DIR), "aligner_rf.joblib")
    db_path = str(Config.DB_PATH)
    clf = joblib.load(model_path)
    
    aligner = MedicalEntityAligner()
    linker = UniversalEntityLinker(db_path=db_path, threshold=0.5)

    # 2. 模拟一个更有挑战性的场景（包含别名、简称、同音干扰）
    test_texts = [
        "患者表现为典型的HTN症状，伴有胸闷，怀疑是CAD引起的心肌梗死。",
        "医生开具了阿斯匹林用于抗凝，患者既往有糖尿病病史。",
        "行胃癌根治术后，切除组织送病理检查。"
    ]

    for text in test_texts:
        print(f"\n[分析文本]: {text}")
        results = predict_text(text, clf, aligner, linker, topk=3)
        
        for r in results:
            print(f"  - Mention: {r['mention']} ({r['span']})")
            if r["linked_entity"]:
                print(f"    → 链接到: {r['linked_entity']['entity_name']} (得分: {r['linked_entity']['score']:.4f})")
            else:
                print(f"    → 链接到: NIL (未找到合适匹配)")

