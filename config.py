import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config: 
    """项目全局配置"""
    
    # 路径配置
    PROJECT_ROOT = Path(__file__).parent.absolute()
    DATA_DIR = PROJECT_ROOT / 'data'
    DB_PATH = PROJECT_ROOT / 'medical_kg. db'
    LOG_DIR = PROJECT_ROOT / 'logs'
    MODEL_DIR = PROJECT_ROOT / 'models'
    
    # 数据集配置
    YIDU_S4K_DIR = os.getenv('YIDU_S4K_PATH', str(DATA_DIR / 'yidu_s4k'))
    
    # 文本处理配置
    JIEBA_DICT_PATH = DATA_DIR / 'medical_terms.txt'
    STOPWORDS_PATH = DATA_DIR / 'stopwords.txt'
    
    # 实体链接配置
    ENTITY_LINKING_THRESHOLD = 0.8
    ENTITY_LINKING_TOP_K = 1
    
    # 实体对齐配置
    ALIGNMENT_THRESHOLD = 0.65
    ALIGNMENT_BLOCKING_ENABLED = True
    
    # 关系抽取配置
    RELATION_TYPES = [
        'causes',        # 导致
        'treats',        # 治疗
        'contraindicated',  # 禁忌
        'complication',  # 并发症
        'symptom',       # 症状
    ]
    
    # 数据库配置
    DB_BATCH_SIZE = 100
    DB_TRANSACTION_ENABLED = True
    
    # 日志配置
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 模型配置
    ML_MODEL_TYPE = 'random_forest'  # 'random_forest', 'svm', 'logistic_regression'
    TRAIN_TEST_SPLIT = 0.8
    
    @classmethod
    def ensure_directories(cls):
        """确保必要的目录存在"""
        for directory in [cls.DATA_DIR, cls.LOG_DIR, cls. MODEL_DIR]:
            directory.mkdir(parents=True, exist_ok=True)