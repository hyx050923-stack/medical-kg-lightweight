#!/usr/bin/env python3
# run.py - 阶段一示例流程（更健壮的实现）

import os
import logging
from logger_config import setup_logging, get_logger
from config import Config
from data_loader import YiduS4KDataLoader
from kg_builder import MedicalKGBuilder
from entity_recognizer import MedicalEntityRecognizer
from entity_linker import UniversalEntityLinker

def main():
    # 初始化目录与日志
    Config.ensure_directories()
    setup_logging()
    logger = get_logger(__name__)
    logger.info("启动阶段一示例流程")

    # 确认数据集路径
    yidu_dir = os.getenv('YIDU_S4K_PATH', str(Config.YIDU_S4K_DIR))
    logger.info(f"Yidu-S4K 数据路径: {yidu_dir}")

    # 初始化数据加载器
    try:
        loader = YiduS4KDataLoader(yidu_dir)
    except FileNotFoundError as e:
        logger.error(f"数据集目录找不到: {e}")
        return

    # 初始化数据库与 KG Builder
    try:
        builder = MedicalKGBuilder(str(Config.DB_PATH), batch_size=Config.DB_BATCH_SIZE)
    except Exception as e:
        logger.error(f"初始化数据库失败: {e}")
        return

    # 从 Yidu-S4K 中抽取实体并批量入库
    logger.info("从数据集中提取实体（可能耗时）...")
    formatted = []  # 确保变量已定义，避免 UnboundLocalError
    try:
        entities = loader.extract_entities_from_task1(limit=None)  # limit 可设置较小做测试
        if not entities:
            logger.warning("未从数据集中提取到实体（entities 为空）。将跳过批量入库步骤。")
        else:
            logger.info(f"提取到 {len(entities)} 个实体样本，开始批量写入数据库...")
            # loader 返回的元素格式：{'name':..., 'type':..., 'aliases':..., 'frequency':...}
            formatted = []
            for e in entities:
                formatted.append({'name': e['name'], 'type': e['type'], 'aliases': e.get('aliases', [])})
            builder.add_entities_batch(formatted)
            logger.info("实体已写入数据库")
    except Exception as e:
        logger.error(f"从数据集中提取/写入实体时发生错误: {e}")

    # 演示：用规则+词典识别文本中的实体，并尝试链接
    sample_text = "患者因胸痛入院，既往高血压史，给予阿司匹林100mg口服。"
    logger.info(f"示例文本: {sample_text}")

    # 如果 formatted 为空，则仍然构造一个最小本地词典以演示识别流程
    recognizer_dict = []
    if formatted:
        recognizer_dict = [{'name': x['name'], 'type': x['type']} for x in formatted[:200]]
    else:
        # 最小词典（演示用）
        recognizer_dict = [
            {'name': '高血压', 'type': 'disease'},
            {'name': '阿司匹林', 'type': 'drug'},
            {'name': '胸痛', 'type': 'symptom'},
        ]
        logger.warning("使用内置最小词典进行演示（因为未导入实体库）")

    recognizer = MedicalEntityRecognizer(medical_dict=recognizer_dict)
    try:
        recognized = recognizer.recognize(sample_text)
    except Exception as e:
        logger.error(f"实体识别失败: {e}")
        recognized = []

    logger.info(f"识别到实体: {recognized}")

    # 实体链接（把识别到的 mention 链接到数据库实体）
    try:
        linker = UniversalEntityLinker(str(Config.DB_PATH), threshold=Config.ENTITY_LINKING_THRESHOLD)
    except Exception as e:
        logger.error(f"初始化实体链接器失败: {e}")
        linker = None

    mentions = [(mention, etype) for (mention, etype, pos) in recognized]
    if linker:
        link_results = linker.link_entities_batch(mentions)
    else:
        link_results = [[] for _ in mentions]

    for mention, results in zip(mentions, link_results):
        logger.info(f"Mention={mention}: Links={results}")

    logger.info("示例流程结束")

if __name__ == '__main__':
    main()