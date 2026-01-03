import numpy as np
import logging
import re
from typing import Tuple, List
from sklearn.preprocessing import MinMaxScaler
from rapidfuzz import fuzz # 建议统一使用 rapidfuzz，速度更快
from pypinyin import lazy_pinyin
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
        self.core_suffixes = ['癌', '瘤', '炎', '术', '病', '综合征', '颗粒', '胶囊', '注射液']

    def get_pinyin(self, text: str) -> str:
        """转换简写拼音，如 '胃癌' -> 'wa'"""
        return "".join(lazy_pinyin(text))
    
    def extract_features(self, e1: dict, e2: dict) -> np.ndarray:
        name1 = e1.get('name', '')
        name2 = e2.get('name', '')
        
        name1_norm = self._normalize_abbreviation(e1['name'].upper())
        name2_norm = self._normalize_abbreviation(e2['name'].upper())
        
        # 1. 基础文本相似度 (Rapidfuzz)
        token_set = fuzz.token_set_ratio(name1_norm, name2_norm) / 100.0
        partial_ratio = fuzz.partial_ratio(name1_norm, name2_norm) / 100.0
        
        # 2. 拼音相似度 (解决同音错字: 阿司匹林 vs 阿斯匹林)
        py1 = self.get_pinyin(name1_norm)
        py2 = self.get_pinyin(name2_norm)
        pinyin_sim = fuzz.ratio(py1, py2) / 100.0
        
        # 3. 医疗核心词匹配特征
        # 如果两个词都包含相同的核心后缀（如都是“癌”），则该特征为 1
        core_match = 0.0
        for s in self.core_suffixes:
            if s in name1_norm and s in name2_norm:
                core_match = 1.0
                break
        
        # 4. 长度与包含特征
        len1, len2 = len(name1_norm), len(name2_norm)
        len_ratio = min(len1, len2) / max(len1, len2, 1)
        is_contained = 1.0 if (name1_norm in name2_norm or name2_norm in name1_norm) else 0.0
        
        # 5. 类型匹配特征
        type_match = 1.0 if e1.get('type') == e2.get('type') else 0.0
        
        #新增"缩写匹配"强特征
        abbrev_match = 1.0 if name1_norm == name2_norm and e1['name'] != e2['name'] else 0.0

        # 构造特征向量 (增加了拼音和核心词维度)
        features = np.array([
            token_set,      # 集合相似度
            partial_ratio,  # 部分匹配得分
            pinyin_sim,     # 拼音相似度 [新增]
            core_match,     # 医疗核心词命中 [新增]
            len_ratio,      # 长度比例
            is_contained,   # 包含关系
            type_match,      # 类型一致性
            abbrev_match
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