# 示例运行脚本：run_stage1.py
# 用途：初始化日志 -> 从 Yidu-S4K 提取实体 -> 批量写入 SQLite -> 用示例文本做识别与实体链接演示

import os
from logger_config import setup_logging, get_logger
from config import Config
from data_loader import YiduS4KDataLoader
from kg_builder import MedicalKGBuilder
from entity_recognizer import MedicalEntityRecognizer
from entity_linker import UniversalEntityLinker

def main():
    # 1. 初始化
    Config.ensure_directories()
    logger = setup_logging()
    logger.info("启动阶段一示例流程")
    
    # 2. 校验数据集路径
    yidu_dir = os.getenv('YIDU_S4K_PATH', Config.YIDU_S4K_DIR)
    logger.info(f"Yidu-S4K 数据路径: {yidu_dir}")
    
    # 3. 加载数据（若路径不存在会报错）
    try:
        loader = YiduS4KDataLoader(yidu_dir)
    except FileNotFoundError as e:
        logger.error(f"数据集目录找不到，请确认 Yidu-S4K 已解压到 {yidu_dir}")
        return
    
    # 4. 初始化数据库与 KG Builder
    builder = MedicalKGBuilder(str(Config.DB_PATH), batch_size=Config.DB_BATCH_SIZE)
    
    # 5. 从 Yidu-S4K 中抽取实体并批量入库
    logger.info("从数据集中提取实体（可能耗时）...")
    entities = loader.extract_entities_from_task1(limit=None)  # limit 可设置较小做测试
    if not entities:
        logger.warning("未提取到实体，请确认数据集格式是否正确")
    else:
        logger.info(f"提取到 {len(entities)} 个实体样本，开始批量写入数据库...")
        # 注意：loader 提取出的 dict 需要 key 与 builder.add_entities_batch 接口匹配
        # loader 返回: {'name': str, 'type': str, 'aliases': [...], 'frequency': int}
        # 转换为 builder 所需格式（name, type, aliases）
        formatted = []
        for e in entities:
            formatted.append({'name': e['name'], 'type': e['type'], 'aliases': e.get('aliases', [])})
        builder.add_entities_batch(formatted)
        logger.info("实体已写入数据库")
    
    # 6. 演示：用规则+词典识别文本中的实体，并尝试链接
    sample_text = "患者因胸痛入院，既往高血压史，给予阿司匹林100mg口服。"
    logger.info(f"示例文本: {sample_text}")
    
    recognizer = MedicalEntityRecognizer(medical_dict=[{'name': x['name'], 'type': x['type']} for x in formatted[:200]])
    recognized = recognizer.recognize(sample_text)
    logger.info(f"识别到实体: {recognized}")
    
    # 7. 实体链接（把识别到的 mention 链接到数据库实体）
    linker = UniversalEntityLinker(str(Config.DB_PATH), threshold=Config.ENTITY_LINKING_THRESHOLD)
    mentions = [(mention, etype) for (mention, etype, pos) in recognized]
    link_results = linker.link_entities_batch(mentions)
    
    for mention, results in zip(mentions, link_results):
        logger.info(f"Mention={mention}: Links={results}")
    
    logger.info("示例流程结束")

if __name__ == '__main__':
    main()