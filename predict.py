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
def predict_text(
    text: str,
    clf,
    aligner,
    linker,
    topk: int = 3,
    min_score: float = 0.4
) -> List[Dict]:
    """
    对一段文本做实体识别 + 实体链接（Ranking）
    """
    recognizer = MedicalEntityRecognizer()
    mentions = recognizer.recognize(text)

    results = []

    for mention_text, mention_type, span in mentions:
        # 1️⃣ Blocking：生成候选实体
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

        # 2️⃣ Ranking
        ranked = rank_entities_for_mention(
            mention_text,
            mention_type,
            candidates,
            clf,
            aligner,
            topk=topk
        )

        best = ranked[0]

        # 3️⃣ 决策（支持 NIL）
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

    model_path = os.path.join(str(Config.MODEL_DIR), "aligner_rf.joblib")
    db_path = str(Config.DB_PATH)

    print(f"[INFO] Loading model from {model_path}")
    clf = joblib.load(model_path)

    aligner = MedicalEntityAligner()
    linker = UniversalEntityLinker(
        db_path=db_path,
        threshold=Config.ENTITY_LINKING_THRESHOLD
    )

    # 🔍 测试文本
    text = "患者因胃癌入院，既往高血压史，行胃癌根治术。"

    print("\n[INPUT TEXT]")
    print(text)

    results = predict_text(
        text,
        clf,
        aligner,
        linker,
        topk=3,
        min_score=0.4
    )

    print("\n[LINK RESULTS]")
    for r in results:
        print("--------------------------------------------------")
        print(f"Mention: {r['mention']}  Span: {r['span']}")
        if r["linked_entity"]:
            print(f"→ Linked: {r['linked_entity']['entity_name']} "
                  f"(score={r['linked_entity']['score']:.3f})")
        else:
            print("→ Linked: NIL")

        print("Candidates:")
        for c in r["candidates"]:
            print(f"   - {c['entity_name']} ({c['entity_type']}) score={c['score']:.3f}")
