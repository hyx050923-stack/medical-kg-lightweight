import sqlite3
import logging
from typing import List, Tuple, Optional, Dict
from fuzzywuzzy import fuzz
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class LinkingResult:
    """实体链接结果"""
    mention:  str  # 提及文本
    entity_id: int  # 链接到的实体ID
    entity_name: str  # 实体规范名称
    entity_type: str  # 实体类型
    confidence: float  # 链接置信度
    alias_used: Optional[str] = None  # 使用的别名

class UniversalEntityLinker:
    """
    通用实体链接器 - 替代硬编码的if判断
    从数据库的 entities + aliases 表批量模糊匹配
    """
    
    def __init__(self, db_path: str, threshold: float = 0.8):
        self.db_path = db_path
        self.threshold = threshold
        self. entity_cache = {}  # 实体缓存（实体类型 -> 实体列表）
        self._load_entities_to_cache()
    
    def _load_entities_to_cache(self):
        """预加载所有实体和别名到内存（避免重复查询）"""
        conn = sqlite3.connect(self. db_path)
        cursor = conn.cursor()
        
        try:
            # 查询所有实体+别名
            cursor.execute("""
                SELECT e.id, e.name, e.entity_type, GROUP_CONCAT(a.alias, '|') as aliases
                FROM entities e
                LEFT JOIN aliases a ON e.id = a. entity_id
                GROUP BY e.id
            """)
            
            for entity_id, entity_name, entity_type, aliases_str in cursor.fetchall():
                if entity_type not in self.entity_cache:
                    self.entity_cache[entity_type] = []
                
                alias_list = aliases_str.split('|') if aliases_str else []
                self.entity_cache[entity_type].append({
                    'id': entity_id,
                    'name': entity_name,
                    'aliases': alias_list,
                    'type': entity_type
                })
            
            logger.info(f"已加载 {sum(len(v) for v in self.entity_cache.values())} 个实体到缓存")
        
        except Exception as e:
            logger. error(f"加载实体缓存失败: {e}")
        finally:
            conn.close()
    
    def link_mention(self, mention: str, mention_type: Optional[str] = None, 
                     top_k: int = 1) -> List[LinkingResult]: 
        """
        链接一个实体提及到知识库
        
        Args: 
            mention: 文本中的实体提及
            mention_type: 提及的实体类型（如不指定则在所有类型中搜索）
            top_k: 返回Top-K个候选
        
        Returns:
            按置信度排序的链接结果列表
        """
        candidates = []
        
        # 确定搜索范围
        search_types = [mention_type] if mention_type else list(self.entity_cache.keys())
        
        for entity_type in search_types:
            entities = self.entity_cache.get(entity_type, [])
            
            for entity in entities:
                # 计算与规范名称的相似度
                name_score = fuzz.token_set_ratio(mention, entity['name']) / 100.0
                
                # 计算与别名的最高相似度
                alias_scores = [
                    (alias, fuzz.token_set_ratio(mention, alias) / 100.0)
                    for alias in entity['aliases']
                ]
                best_alias_score = max(alias_scores, key=lambda x: x[1], default=(None, 0))
                
                # 取最高分数
                if best_alias_score[1] > name_score:
                    score = best_alias_score[1]
                    alias_used = best_alias_score[0]
                else:
                    score = name_score
                    alias_used = None
                
                if score >= self.threshold:
                    candidates.append(LinkingResult(
                        mention=mention,
                        entity_id=entity['id'],
                        entity_name=entity['name'],
                        entity_type=entity['type'],
                        confidence=score,
                        alias_used=alias_used
                    ))
        
        # 按置信度降序排序，返回Top-K
        candidates.sort(key=lambda x: x.confidence, reverse=True)
        return candidates[:top_k]
    
    def link_entities_batch(self, mentions: List[Tuple[str, Optional[str]]]) -> List[List[LinkingResult]]:
        """
        批量链接多个实体提及
        
        Args: 
            mentions: 列表，每项为 (mention_text, entity_type) 元组
        
        Returns: 
            链接结果列表
        """
        results = []
        for mention, entity_type in mentions:
            try:
                linking_results = self.link_mention(mention, entity_type, top_k=1)
                results.append(linking_results)
            except Exception as e:
                logger.error(f"链接 '{mention}' 失败: {e}")
                results.append([])
        
        return results
    
    def add_entity_to_cache(self, entity_id: int, name: str, entity_type: str, aliases: List[str]):
        """运行时添加新实体到缓存"""
        if entity_type not in self.entity_cache:
            self.entity_cache[entity_type] = []
        
        self.entity_cache[entity_type].append({
            'id': entity_id,
            'name': name,
            'aliases': aliases,
            'type': entity_type
        })

    def get_candidates(self, mention_text, topn=50):
        scored = []
        for e in self.entities:   # 855 个实体
            score = fuzz.partial_ratio(mention_text, e["name"])
            scored.append((score, e))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:topn]]
