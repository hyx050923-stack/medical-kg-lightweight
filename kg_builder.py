import sqlite3
import logging
from typing import List, Dict, Optional, Tuple
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)

class DatabaseConnection:
    """数据库连接单例（避免多连接冲突）"""
    _instance = None
    _db_path = None
    
    def __new__(cls, db_path: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            if db_path:
                cls._db_path = db_path
                cls._instance._init_db()
        return cls._instance
    
    def _init_db(self):
        """初始化数据库连接"""
        self.conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        logger.info(f"数据库连接已建立:  {self._db_path}")
    
    def get_connection(self):
        return self.conn
    
    def close(self):
        if self.conn:
            self. conn.close()
            logger. info("数据库连接已关闭")


class MedicalKGBuilder:
    """
    医疗知识图谱构建器 - 支持事务、批量操作、异常处理
    """
    
    # 数据库表定义
    SCHEMAS = {
        'entities': """
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                entity_type TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(entity_type);
            CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name);
        """,
        'aliases': """
            CREATE TABLE IF NOT EXISTS aliases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id INTEGER NOT NULL,
                alias TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_alias ON aliases(alias);
        """,
        'relationships': """
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_entity_id INTEGER NOT NULL,
                relation_type TEXT NOT NULL,
                target_entity_id INTEGER NOT NULL,
                confidence REAL DEFAULT 0.5,
                evidence TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_entity_id) REFERENCES entities(id) ON DELETE CASCADE,
                FOREIGN KEY (target_entity_id) REFERENCES entities(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_relation_type ON relationships(relation_type);
            CREATE INDEX IF NOT EXISTS idx_source_entity ON relationships(source_entity_id);
        """,
    }
    
    def __init__(self, db_path: str, batch_size: int = 100):
        """
        Args: 
            db_path: SQLite数据库路径
            batch_size: 批量操作时的批次大小
        """
        self.db_path = db_path
        self.batch_size = batch_size
        self.db_conn = DatabaseConnection(db_path)
        self.conn = self.db_conn.get_connection()
        self._init_tables()
        
        # 批量操作缓冲区
        self.entity_buffer = []
        self.alias_buffer = []
        self.relationship_buffer = []
    
    def _init_tables(self):
        """初始化数据库表"""
        cursor = self.conn.cursor()
        try:
            for table_name, schema in self.SCHEMAS.items():
                cursor.execute(schema)
            self.conn.commit()
            logger.info("数据库表初始化成功")
        except sqlite3.Error as e:
            logger.error(f"数据库初始化失败: {e}")
            self.conn.rollback()
    
    @contextmanager
    def transaction(self):
        """事务上下文管理器"""
        try:
            yield
            self.conn.commit()
            logger.debug("事务提交成功")
        except sqlite3.Error as e:
            self.conn.rollback()
            logger.error(f"事务执行失败: {e}")
            raise
    
    def add_entity(self, name: str, entity_type: str, aliases: List[str] = None) -> Optional[int]:
        """
        添加单个实体
        
        Args:
            name: 实体规范名称
            entity_type: 实体类型
            aliases: 别名列表
        
        Returns:
            实体ID（失败返回None）
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO entities (name, entity_type) VALUES (?, ?)",
                (name, entity_type)
            )
            entity_id = cursor.lastrowid
            
            # 添加别名
            if aliases:
                for alias in aliases:
                    try:
                        cursor.execute(
                            "INSERT OR IGNORE INTO aliases (entity_id, alias) VALUES (?, ?)",
                            (entity_id, alias)
                        )
                    except sqlite3.IntegrityError:
                        logger.warning(f"别名已存在: {alias}")
            
            self.conn.commit()
            logger.debug(f"已添加实体: {name} (ID: {entity_id})")
            return entity_id
        
        except sqlite3.Error as e:
            logger.error(f"添加实体失败 ({name}): {e}")
            self.conn.rollback()
            return None
    
    def add_entities_batch(self, entities: List[Dict]) -> List[int]:
        """
        批量添加实体（优化性能）
        
        Args:
            entities: [{'name': str, 'type': str, 'aliases': [str]}, ...]
        
        Returns:
            实体ID列表
        """
        entity_ids = []
        
        try:
            cursor = self.conn.cursor()
            
            for entity in entities:
                name = entity.get('name')
                entity_type = entity. get('type')
                aliases = entity.get('aliases', [])
                
                if not name or not entity_type:
                    continue
                
                # 插入实体
                cursor.execute(
                    "INSERT OR REPLACE INTO entities (name, entity_type) VALUES (?, ?)",
                    (name, entity_type)
                )
                entity_id = cursor.lastrowid
                entity_ids.append(entity_id)
                
                # 插入别名
                for alias in aliases:
                    cursor.execute(
                        "INSERT OR IGNORE INTO aliases (entity_id, alias) VALUES (?, ?)",
                        (entity_id, alias)
                    )
            
            self.conn.commit()
            logger.info(f"批量添加 {len(entity_ids)} 个实体成功")
        
        except sqlite3.Error as e:
            logger.error(f"批量添加实体失败:  {e}")
            self.conn.rollback()
        
        return entity_ids
    
    def add_relationship(self, source_entity_id: int, relation_type: str, 
                        target_entity_id: int, confidence: float = 0.5, 
                        evidence: str = None) -> Optional[int]:
        """
        添加单个关系
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """INSERT INTO relationships 
                   (source_entity_id, relation_type, target_entity_id, confidence, evidence)
                   VALUES (?, ?, ?, ?, ?)""",
                (source_entity_id, relation_type, target_entity_id, confidence, evidence)
            )
            relationship_id = cursor.lastrowid
            self.conn.commit()
            logger.debug(f"已添加关系: {source_entity_id} --{relation_type}--> {target_entity_id}")
            return relationship_id
        
        except sqlite3.Error as e:
            logger.error(f"添加关系失败:  {e}")
            self.conn.rollback()
            return None
    
    def add_relationships_batch(self, relationships: List[Dict]) -> List[int]:
        """
        批量添加关系
        
        Args:
            relationships: [{'source_id': int, 'type': str, 'target_id': int, 'confidence': float}, ...]
        """
        relationship_ids = []
        
        try:
            cursor = self.conn.cursor()
            
            for rel in relationships:
                source_id = rel.get('source_id')
                rel_type = rel.get('type')
                target_id = rel.get('target_id')
                confidence = rel.get('confidence', 0.5)
                evidence = rel.get('evidence')
                
                if not (source_id and rel_type and target_id):
                    continue
                
                cursor.execute(
                    """INSERT INTO relationships 
                       (source_entity_id, relation_type, target_entity_id, confidence, evidence)
                       VALUES (?, ?, ?, ?, ?)""",
                    (source_id, rel_type, target_id, confidence, evidence)
                )
                relationship_ids.append(cursor.lastrowid)
            
            self.conn.commit()
            logger.info(f"批量添加 {len(relationship_ids)} 个关系成功")
        
        except sqlite3.Error as e:
            logger.error(f"批量添加关系失败: {e}")
            self.conn.rollback()
        
        return relationship_ids
    
    def query_entity(self, entity_name: str) -> Optional[Dict]:
        """查询实体"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT * FROM entities WHERE name = ?",
                (entity_name,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
        except sqlite3.Error as e:
            logger.error(f"查询实体失败: {e}")
            return None
    
    def get_entity_relationships(self, entity_id: int, relation_type: str = None) -> List[Dict]:
        """获取实体的关系"""
        try:
            cursor = self.conn.cursor()
            
            if relation_type:
                cursor. execute(
                    """SELECT r.*, e.name as target_name FROM relationships r
                       JOIN entities e ON r.target_entity_id = e.id
                       WHERE r.source_entity_id = ?  AND r.relation_type = ? """,
                    (entity_id, relation_type)
                )
            else:
                cursor.execute(
                    """SELECT r.*, e.name as target_name FROM relationships r
                       JOIN entities e ON r.target_entity_id = e.id
                       WHERE r.source_entity_id = ?""",
                    (entity_id,)
                )
            
            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"查询关系失败: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """获取知识图谱统计信息"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM entities")
            entity_count = cursor.fetchone()[0]
            
            cursor. execute("SELECT COUNT(*) FROM relationships")
            relationship_count = cursor.fetchone()[0]
            
            cursor.execute("""SELECT entity_type, COUNT(*) FROM entities 
                            GROUP BY entity_type""")
            type_distribution = dict(cursor.fetchall())
            
            cursor.execute("""SELECT relation_type, COUNT(*) FROM relationships 
                            GROUP BY relation_type""")
            relation_distribution = dict(cursor.fetchall())
            
            return {
                'total_entities': entity_count,
                'total_relationships': relationship_count,
                'entity_type_distribution': type_distribution,
                'relation_type_distribution':  relation_distribution,
            }
        except sqlite3.Error as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}