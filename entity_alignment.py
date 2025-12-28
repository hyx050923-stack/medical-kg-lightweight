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
    
    def extract_features(self, entity1: dict, entity2: dict) -> np.ndarray:
        """
        提取实体对的特征向量 (维度=8)
        
        特征1: 字符串相似度 (Token Set Ratio)
        特征2: 字符串相似度 (Partial Ratio)
        特征3: 实体类型匹配度 (0/1)
        特征4: 缩写归一化后的相似度
        特征5: 数字单位匹配度
        特征6: 长度差异惩罚 (归一化)
        特征7: 共享关键词数量
        特征8: 医学同义词库匹配度
        """
        features = []
        
        name1, type1 = entity1['name'], entity1['type']
        name2, type2 = entity2['name'], entity2['type']
        
        # 特征1: Token Set Ratio
        f1 = fuzz.token_set_ratio(name1, name2) / 100.0
        features.append(f1)
        
        # 特征2: Partial Ratio
        f2 = fuzz.partial_ratio(name1, name2) / 100.0
        features.append(f2)
        
        # 特征3: 实体类型匹配
        f3 = 1.0 if type1 == type2 else 0.0
        features.append(f3)
        
        # 特征4: 缩写归一化相似度
        norm1 = self._normalize_abbreviation(name1)
        norm2 = self._normalize_abbreviation(name2)
        f4 = fuzz.token_set_ratio(norm1, norm2) / 100.0 if norm1 != name1 or norm2 != name2 else 0.0
        features. append(f4)
        
        # 特征5: 数字单位匹配
        f5 = self._match_numeric_units(name1, name2)
        features.append(f5)
        
        # 特征6: 长度差异惩罚
        len_diff = abs(len(name1) - len(name2)) / max(len(name1), len(name2))
        f6 = 1.0 - min(len_diff, 0.5)  # 最多惩罚0.5
        features.append(f6)
        
        # 特征7: 共享关键词
        tokens1 = set(name1.split())
        tokens2 = set(name2.split())
        shared = len(tokens1 & tokens2)
        f7 = shared / max(len(tokens1), len(tokens2)) if max(len(tokens1), len(tokens2)) > 0 else 0.0
        features. append(f7)
        
        # 特征8: 实体类型不兼容惩罚
        types_key = tuple(sorted([type1, type2]))
        f8 = 0.0 if self. ENTITY_TYPE_INCOMPATIBLE. get(types_key, True) else 0.3
        features.append(f8)
        
        return np. array(features)
    
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