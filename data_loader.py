import json
import logging
from typing import List, Dict, Tuple, Iterator
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

class YiduS4KDataLoader: 
    """
    Yidu-S4K 医疗知识图谱数据集加载器
    支持：任务1（实体识别）和任务2（实体属性提取）
    """
    
    # 实体类型映射（根据 Yidu-S4K 标准）
    ENTITY_TYPES = {
        '疾病和诊断': 'disease',
        '症状':  'symptom',
        '药物': 'drug',
        '医学检查': 'examination',
        '医学处置': 'treatment',
        '身体部位': 'body_part',
        '医学属性': 'attribute',
    }
    
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir:  Yidu-S4K 数据集目录
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据集目录不存在: {data_dir}")
    
    def load_task1_training(self, split: str = 'all') -> Iterator[Dict]:
        """
        加载任务1训练数据（实体识别）
        
        Args:
            split: 'part1', 'part2', 或 'all'
        
        Yields:
            {'text': str, 'entities': [{'offset': (start, end), 'type': str, 'text': str}, ...]}
        """
        if split in ['part1', 'all']:
            yield from self._load_jsonl(
                self.data_dir / 'subtask1_training_part1.txt'
            )
        
        if split in ['part2', 'all']:
            yield from self._load_jsonl(
                self.data_dir / 'subtask1_training_part2.txt'
            )
    
    def load_task1_test(self) -> List[Dict]:
        """
        加载任务1测试数据（带标准答案）
        
        Returns: 
            [{'originalText': str, 'entities': [{'start_pos': int, 'end_pos': int, 
                                                   'entity_label': str, 'entity_text': str}, ...]}, ...]
        """
        test_file = self.data_dir / 'subtask1_test_set_with_answer.json'
        
        if not test_file.exists():
            logger.warning(f"测试集文件不存在: {test_file}")
            return []
        
        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def load_task2_training(self, split: str = 'all') -> pd.DataFrame:
        """
        加载任务2训练数据（实体属性提取）
        
        Args: 
            split: 'part1', 'part2', 或 'all'
        
        Returns:
            pandas DataFrame
        """
        dfs = []
        
        if split in ['part1', 'all']:
            df_part1 = pd.read_excel(
                self.data_dir / 'subtask2_training_part1.xlsx',
                engine='openpyxl'
            )
            dfs.append(df_part1)
        
        if split in ['part2', 'all']:
            df_part2 = pd. read_excel(
                self. data_dir / 'subtask2_training_part2.xlsx',
                engine='openpyxl'
            )
            dfs.append(df_part2)
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()
    
    def load_task2_test(self) -> pd.DataFrame:
        """
        加载任务2测试数据
        """
        test_file = self. data_dir / 'subtask2_test. xlsx'
        
        if not test_file.exists():
            logger.warning(f"测试集文件不存在: {test_file}")
            return pd.DataFrame()
        
        return pd.read_excel(test_file, engine='openpyxl')
    
    @staticmethod
    def _load_jsonl(file_path: Path) -> Iterator[Dict]:
        """加载JSONL格式文件"""
        if not file_path.exists():
            logger.warning(f"文件不存在: {file_path}")
            return
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    yield item
                except json.JSONDecodeError as e:
                    logger.error(f"JSON解析错误（第 {line_num} 行）: {e}")
    
    def extract_entities_from_task1(self, limit: int = None) -> List[Dict]:
        """
        从任务1数据中提取所有实体（用于初始化实体库）
        
        Args: 
            limit: 最多提取数量
        
        Returns: 
            [{'name': str, 'type': str, 'aliases': [str]}, ...]
        """
        entities = {}  # name -> {'type': str, 'aliases':  set()}
        count = 0
        
        for record in self.load_task1_training():
            for entity in record.get('entities', []):
                # Yidu-S4K 格式:  offset, type, text
                entity_name = entity. get('text', '')
                entity_type = entity.get('type', '')
                
                if not entity_name or not entity_type:
                    continue
                
                # 规范化实体类型
                normalized_type = self. ENTITY_TYPES.get(entity_type, 'unknown')
                
                if entity_name not in entities:
                    entities[entity_name] = {
                        'type': normalized_type,
                        'aliases': set(),
                        'frequency': 0
                    }
                
                entities[entity_name]['frequency'] += 1
                
                count += 1
                if limit and count >= limit:
                    break
            
            if limit and count >= limit:
                break
        
        # 转换为列表
        result = [
            {
                'name': name,
                'type': info['type'],
                'aliases':  list(info['aliases']),
                'frequency': info['frequency']
            }
            for name, info in entities.items()
        ]
        
        logger.info(f"从 Yidu-S4K 提取 {len(result)} 个唯一实体")
        return result