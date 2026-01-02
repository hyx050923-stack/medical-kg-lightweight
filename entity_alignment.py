import numpy as np
import logging
from typing import Tuple, List
from sklearn.preprocessing import MinMaxScaler
from fuzzywuzzy import fuzz

logger = logging.getLogger(__name__)

class MedicalEntityAligner:
    """
    医疗实体对齐器 - 补充医疗领域强区分特征
    特征：字符相似度 + 实体类型匹配 + 缩写归一化 + 数字单位匹配
    """
    
    # 医疗实体类型的互斥关系
    ENTITY_TYPE_INCOMPATIBLE = {
        ('disease', 'drug'): True,
        ('disease', 'symptom'): False,  # 症状可伴随疾病
        ('drug', 'symptom'): True,
        ('drug', 'treatment'): False,   # 药物是治疗方式
    }
    
    # 常见医疗缩写及其展开
    ABBREVIATION_MAP = {
        'HTN': '高血压',
        'DM': '糖尿病',
        'CAD': '冠心病',
        'COPD': '慢性阻塞性肺病',
        'MI': '心肌梗死',
        'PCI': '经皮冠状动脉介入',
        'CVD': '心血管疾病',
        'CHF': '充血性心力衰竭',
        'MACE': '主要不良心脑血管事件',
        'BP': '血压',
        'HR': '心率',
        'RR': '呼吸频率',
        'BNP': 'B型利钠肽',
        'ECG': '心电图',
        'CT': '计算机断层扫描',
        'MRI': '磁共振成像',
    }
    
    def __init__(self):
        self.scaler = MinMaxScaler()
    
    def extract_features(self, e1: dict, e2: dict) -> np.ndarray:
        """
        e1: {'name': mention_text, 'type': mention_type}
        e2: {'name': entity_name, 'type': entity_type}
        """

        name1 = e1['name']
        name2 = e2['name']

        # ---------- 基础文本特征（削弱极端值） ----------
        token_set = fuzz.token_set_ratio(name1, name2) / 100.0
        token_sort = fuzz.token_sort_ratio(name1, name2) / 100.0
        partial = fuzz.partial_ratio(name1, name2) / 100.0

        # clip，防止 1.0 成为完美信号
        token_set = min(token_set, 0.95)
        token_sort = min(token_sort, 0.95)
        partial = min(partial, 0.95)

        # ---------- 长度特征 ----------
        len1 = len(name1)
        len2 = len(name2)
        len_diff = abs(len1 - len2) / max(len1, len2, 1)
        len_ratio = min(len1, len2) / max(len1, len2, 1)

        # ---------- 字符级重叠 ----------
        set1 = set(name1)
        set2 = set(name2)
        char_jaccard = len(set1 & set2) / max(len(set1 | set2), 1)

        # ---------- 类型特征 ----------
        same_type = int(e1.get('type') == e2.get('type'))

        # ---------- 包含关系（弱信号） ----------
        contain = int(name1 in name2 or name2 in name1)

        # ---------- 正则化 exact match（关键！） ----------
        exact_match = int(name1 == name2)
        exact_match_soft = exact_match * 0.1  # 只能是弱提示

        features = np.array([
            token_set,
            token_sort,
            partial,
            char_jaccard,
            len_diff,
            len_ratio,
            same_type,
            contain,
            exact_match_soft
        ], dtype=np.float32)

        return features
    
    @staticmethod
    def _normalize_abbreviation(text: str) -> str:
        """缩写规范化"""
        result = text
        for abbr, expansion in MedicalEntityAligner. ABBREVIATION_MAP.items():
            result = result.replace(abbr, expansion)
        return result
    
    @staticmethod
    def _match_numeric_units(text1: str, text2: str) -> float:
        """
        数字单位匹配度
        例如:  "阿司匹林100mg" vs "阿司匹林" 应该有较高相似度
        """
        import re
        
        # 提取数字和单位
        nums1 = re.findall(r'(\d+\. ?\d*)\s*([a-zA-Z%]+)?', text1)
        nums2 = re.findall(r'(\d+\.?\d*)\s*([a-zA-Z%]+)?', text2)
        
        if not nums1 or not nums2:
            return 0.0
        
        # 简单启发式：如果单位相同，分数+0.3
        units1 = set(u[1] for u in nums1 if u[1])
        units2 = set(u[1] for u in nums2 if u[1])
        
        if units1 & units2:
            return 0.3
        
        return 0.0
    
    def classify_alignment(self, features: np.ndarray, threshold: float = 0.65) -> Tuple[bool, float]:
        """
        根据特征向量判断两个实体是否应该对齐
        
        Returns:
            (是否对齐, 对齐置信度)
        """
        # 简单的加权平均（权重突出字符相似度和类型匹配）
        weights = np.array([0.25, 0.20, 0.20, 0.15, 0.10, 0.05, 0.03, 0.02])
        
        alignment_score = np.dot(features, weights)
        
        is_aligned = alignment_score >= threshold
        
        return is_aligned, alignment_score