import re
import jieba
import logging
from typing import List, Set

logger = logging.getLogger(__name__)

class MedicalTextPreprocessor:
    """医疗文本预处理器，覆盖：全半角转换、标点规范化、特殊字符清理"""
    
    def __init__(self, custom_dict_path:  str = None):
        # 加载医疗词典（支持jieba分词优化）
        if custom_dict_path:
            jieba.load_userdict(custom_dict_path)
        
        self.punctuation_map = self._build_punctuation_map()
    
    @staticmethod
    def _build_punctuation_map() -> dict:
        """全半角标点符号映射表"""
        return {
            '，': ',', '。': '.', '；': ';', '：': ':', '？': '? ',
            '！': '! ', '（': '(', '）': ')', '【': '[', '】': ']',
            '「': '"', '」': '"', '『': "'", '』': "'", '、': ',',
            '\u3000': ' ',  # 全角空格
        }
    
    def normalize_text(self, text: str) -> str:
        """
        医疗文本规范化（多步骤）
        1. 全半角转换
        2. 标点符号统一
        3. 冗余空格清理
        4. 特殊医疗符号处理
        """
        if not text or not isinstance(text, str):
            return ""
        
        # 全半角转换
        text = self._convert_fullwidth(text)
        
        # 标点符号映射
        for cn_punc, en_punc in self.punctuation_map.items():
            text = text.replace(cn_punc, en_punc)
        
        # 冗余空格清理（保留单空格）
        text = re.sub(r'\s+', ' ', text)
        
        # 医疗特殊符号处理（如 "/" "-" 作为范围符号）
        text = re.sub(r'([0-9])\s*[-~]\s*([0-9])', r'\1-\2', text)
        
        # 首尾空格清理
        text = text.strip()
        
        return text
    
    @staticmethod
    def _convert_fullwidth(text: str) -> str:
        """全角字符转半角"""
        result = []
        for char in text:
            code = ord(char)
            if 0xFF01 <= code <= 0xFF5E:  # 全角ASCII范围
                result.append(chr(code - 0xFEE0))
            elif code == 0x3000:  # 全角空格
                result.append(' ')
            else:
                result. append(char)
        return ''.join(result)
    
    def tokenize(self, text: str) -> List[str]:
        """分词（使用jieba）"""
        text = self.normalize_text(text)
        tokens = list(jieba.cut(text))
        return [t for t in tokens if t. strip()]  # 去除空token
    
    def remove_stopwords(self, tokens: List[str], stopwords: Set[str]) -> List[str]:
        """移除停用词"""
        return [t for t in tokens if t not in stopwords]