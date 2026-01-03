import joblib
import os
import numpy as np
from config import Config
from entity_alignment import MedicalEntityAligner
from entity_linker import UniversalEntityLinker

def debug_single_case(mention, target_name_guess=None):
    print(f"\n{'='*20} 诊断 Mention: [{mention}] {'='*20}")
    
    # 1. 加载组件
    db_path = str(Config.DB_PATH)
    model_path = os.path.join(str(Config.MODEL_DIR), "aligner_rf.joblib")
    
    try:
        clf = joblib.load(model_path)
        aligner = MedicalEntityAligner()
        linker = UniversalEntityLinker(db_path=db_path)
    except Exception as e:
        print(f"加载失败: {e}")
        return

    # 2. 检查 Linker 召回情况 (Blocking)
    print(f"[1] 正在召回候选 (get_candidates)...")
    # 强制打印前 5 个，不管分数多低
    candidates = linker.get_candidates(mention, max_candidates=5)
    
    if not candidates:
        print("❌ 致命问题：没有召回到任何候选实体！")
        print("   -> 请检查数据库 aliases 表是否有该缩写")
        print("   -> 请检查 get_candidates 的 fuzzy 匹配逻辑")
        return

    found_target = False
    for i, cand in enumerate(candidates):
        print(f"   候选 #{i+1}: {cand['name']} (ID: {cand['id']}, Linker Score: {cand['score']})")
        if target_name_guess and target_name_guess in cand['name']:
            found_target = True

    if target_name_guess and not found_target:
        print(f"⚠️ 警告：预期目标 '{target_name_guess}' 未出现在前 5 名候选中！")

    # 3. 检查 Aligner 特征与打分 (Ranking)
    print(f"\n[2] 特征提取与模型打分详情...")
    print(f"{'Candidate':<15} | {'Norm':<10} | {'Abbr':<5} | {'Pinyin':<5} | {'Token':<5} | {'Model Score (Prob)'}")
    print("-" * 80)
    
    for cand in candidates:
        # 提取特征
        feat = aligner.extract_features(
            {"name": mention, "type": "unknown"},
            {"name": cand['name'], "type": cand['type']}
        )
        # 获取第 8 维特征 (Abbrev Match) 和 第 1 维 (Token Set)
        # 注意：feat 是 numpy array
        abbrev_feat = feat[7] if len(feat) > 7 else -1
        token_feat = feat[0]
        pinyin_feat = feat[2]
        
        # 预测概率
        prob = clf.predict_proba([feat])[0][1]
        
        # 归一化后的名字
        norm_name = aligner._normalize_abbreviation(mention.upper())
        
        print(f"{cand['name']:<15} | {norm_name:<10} | {abbrev_feat:.1f}   | {pinyin_feat:.1f}   | {token_feat:.1f}   | {prob:.5f}")

if __name__ == "__main__":
    # 诊断 HTN (预期链接到 高血压)
    debug_single_case("HTN", "高血压")
    
    # 诊断 心肌梗死 (预期链接到 急性心肌梗死)
    debug_single_case("心肌梗死", "心肌梗死")