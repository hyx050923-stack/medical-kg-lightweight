import logging
from typing import List, Tuple, Dict
import re

logger = logging.getLogger(__name__)

class MedicalEntityRecognizer:
    """
    医疗命名实体识别器
    使用规则 + 词典 + 可选的ML模型
    """
    
    # 医疗实体的规则模式
    RULE_PATTERNS = {
        'disease': [
            r'[^，。！？;]*?(炎|癌|病|症|梗|衰|综合征|损伤|畸形|缺陷)\b',  # 疾病后缀
            r'(高血压|糖尿病|冠心病|心肌梗死|肺炎|肝炎|肾炎|胃炎|肠炎)',  # 常见疾病
        ],
        'drug': [
            r'(阿司匹林|布洛芬|头孢|青霉素|利尿|他汀|普利|受体阻滞剂)',  # 常见药物
            r'\b\w+(片|胶囊|注射液|口服液|软膏|乳膏)\b',  # 药物剂型
        ],
        'symptom': [
            r'(头痛|胸痛|腹痛|咳嗽|发热|发烧|乏力|疲劳|肿胀|出血)',  # 常见症状
            r'(高血糖|低血糖|高血压|低血压)\b',
        ],
        'examination': [
            r'(心电图|脑电图|CT|MRI|超声|X光|血常规|血生化|尿常规)',
            r'(B超|彩超|核磁共振|内镜|胃镜|肠镜)\b',
        ],
        'treatment': [
            r'(手术|开刀|切除|修补|植入|取出|冲洗|引流|放疗|化疗)',
            r'(穿刺|搭桥|搭建|静脉|输液|注射)\b',
        ],
    }
    
    def __init__(self, medical_dict: List[Dict] = None):
        """
        Args:
            medical_dict: 医疗实体词典，格式:  [{'name': str, 'type': str}, ...]
        """
        self.medical_dict = {}
        if medical_dict: 
            for item in medical_dict:
                name = item['name']
                entity_type = item['type']
                self.medical_dict[name] = entity_type
    
    def recognize(self, text: str, use_rules: bool = True, 
                  use_dict: bool = True) -> List[Tuple[str, str, Tuple[int, int]]]:
        """
        识别文本中的医疗实体
        
        Args:
            text: 输入文本
            use_rules: 是否使用规则匹配
            use_dict:  是否使用词典匹配
        
        Returns: 
            [(entity_text, entity_type, (start_pos, end_pos)), ...]
        """
        entities = []
        
        if use_dict:
            entities. extend(self._dict_based_recognition(text))
        
        if use_rules:
            entities. extend(self._rule_based_recognition(text))
        
        # 去重和排序
        entities = list(set(entities))
        entities.sort(key=lambda x: x[2][0])  # 按位置排序
        
        return entities
    
    def _dict_based_recognition(self, text: str) -> List[Tuple[str, str, Tuple[int, int]]]:
        """基于词典的实体识别"""
        entities = []
        
        # 按长度降序排序（避免子串覆盖）
        sorted_terms = sorted(self.medical_dict.keys(), key=len, reverse=True)
        
        covered_positions = set()  # 已覆盖的字符位置
        
        for term in sorted_terms:
            start = 0
            while True: 
                pos = text.find(term, start)
                if pos == -1:
                    break
                
                end = pos + len(term)
                
                # 检查是否与已有实体重叠
                if not any(covered_positions & set(range(pos, end))):
                    entity_type = self.medical_dict[term]
                    entities.append((term, entity_type, (pos, end)))
                    covered_positions. update(range(pos, end))
                
                start = pos + 1
        
        return entities
    
    def _rule_based_recognition(self, text: str) -> List[Tuple[str, str, Tuple[int, int]]]:
        """基于规则的实体识别"""
        entities = []
        
        for entity_type, patterns in self.RULE_PATTERNS.items():
            for pattern in patterns:
                try:
                    for match in re.finditer(pattern, text):
                        mention = match.group(0)
                        start_pos = match.start()
                        end_pos = match.end()
                        entities.append((mention, entity_type, (start_pos, end_pos)))
                except re.error as e:
                    logger.error(f"正则表达式错误 ({entity_type}): {e}")
        
        return entities
    
    def add_entities_to_dict(self, entities: List[Dict]):
        """运行时添加实体到词典"""
        for entity in entities:
            name = entity. get('name', '')
            entity_type = entity.get('type', 'unknown')
            self.medical_dict[name] = entity_type
            logger.debug(f"添加实体到词典: {name} ({entity_type})")