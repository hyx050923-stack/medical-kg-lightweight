# data_loader.py
import json
import logging
from typing import Iterator, Dict, List
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class YiduS4KDataLoader:
    """
    Yidu-S4K 医疗知识图谱数据集加载器
    - 支持任务1（JSONL）与任务2（Excel）
    - 对 JSONL 使用 utf-8-sig，跳过空行与解析错误
    """
    # 实体类型映射（如需扩展请修改）
    ENTITY_TYPES = {
        '疾病和诊断': 'disease',
        '症状': 'symptom',
        '药物': 'drug',
        '医学检查': 'examination',
        '医学处置': 'treatment',
        '身体部位': 'body_part',
        '医学属性': 'attribute',
    }

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        print("[DEBUG] YiduS4KDataLoader data_dir =", self.data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据集目录不存在: {data_dir}")

    def _load_jsonl(self, file_path: Path) -> Iterator[Dict]:
        """加载 JSONL 格式文件，支持 BOM，并对错误行跳过记录"""
        if not file_path.exists():
            logger.warning(f"文件不存在: {file_path}")
            return

        # 使用 utf-8-sig 去掉可能的 BOM
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    # 跳过空行
                    continue
                try:
                    item = json.loads(line)
                    yield item
                except json.JSONDecodeError as e:
                    logger.error(f"JSON解析错误（第 {line_num} 行）: {e}")
                    # 跳过该行，继续解析下一行
                    continue

    def load_task1_training(self, split: str = 'all') -> Iterator[Dict]:
        """
        加载任务1训练数据（实体识别）:
        - split: 'part1', 'part2', 或 'all'
        Yidu-S4K 的训练文件通常是 JSONL，每行一个 JSON 对象。
        """
        # 如果存在 txt 训练集，优先用 txt
        txt1 = os.path.join(self.data_dir, "subtask1_training_part1.txt")
        if os.path.exists(txt1):
            yield from self.load_task1_training_from_txt()
            return
        if split in ['part1', 'all']:
            p1 = self.data_dir / 'subtask1_training_part1.txt'
            yield from self._load_jsonl(p1)

        if split in ['part2', 'all']:
            p2 = self.data_dir / 'subtask1_training_part2.txt'
            yield from self._load_jsonl(p2)

    def load_task1_test(self) -> List[Dict]:
        """加载任务1测试集（带答案的 JSON 文件）"""
        test_file = self.data_dir / 'subtask1_test_set_with_answer.json'
        if not test_file.exists():
            logger.warning(f"测试集文件不存在: {test_file}")
            return []
        try:
            with open(test_file, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"读取测试集失败: {e}")
            return []

    def extract_entities_from_task1(self, limit: int = None) -> List[Dict]:
        """
        从任务1数据中提取所有实体（用于初始化实体库）
        返回列表 [{'name': str, 'type': str, 'aliases': [], 'frequency': int}, ...]
        """
        entities = {}  # name -> {'type': str, 'aliases': set(), 'frequency': int}
        count = 0

        for record in self.load_task1_training(split='all'):
            # Yidu JSON 格式可能用 'originalText' 或 'text'，兼容处理
            text = record.get('originalText') or record.get('text') or ''
            for ent in record.get('entities', []):
                # Yidu 实体字段常见键： 'text' (实体文本), 'type' 或 'entity_label'
                entity_text = ent.get('text') or ent.get('entity_text') or ''
                entity_type_raw = ent.get('type') or ent.get('entity_label') or ''
                if not entity_text:
                    continue
                normalized_type = self.ENTITY_TYPES.get(entity_type_raw, 'unknown')
                if entity_text not in entities:
                    entities[entity_text] = {'type': normalized_type, 'aliases': set(), 'frequency': 0}
                entities[entity_text]['frequency'] += 1
                count += 1
                if limit and count >= limit:
                    break
            if limit and count >= limit:
                break

        result = []
        for name, info in entities.items():
            result.append({
                'name': name,
                'type': info['type'],
                'aliases': list(info['aliases']),
                'frequency': info['frequency']
            })
        logger.info(f"从 Yidu-S4K 提取 {len(result)} 个唯一实体 (遍历条数: {count})")
        return result
    
    # data_loader.py 中新增
    def load_task1_training_from_txt(self):
        """
        解析 Yidu-S4K Task1 的 subtask1_training_part*.txt（JSONL 格式）
        输出统一格式：
        {
            "text": 原文,
            "entities": [
                {"text": 实体文本, "type": 实体类型}
            ]
        }
        """
        files = [
            "subtask1_training_part1.txt",
            "subtask1_training_part2.txt"
        ]

        for fname in files:
            path = os.path.join(self.data_dir, fname)
            if not os.path.exists(path):
                continue

            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        sample = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    text = sample.get("originalText", "")
                    entities = []

                    for ent in sample.get("entities", []):
                        start = ent.get("start_pos")
                        end = ent.get("end_pos")
                        label = ent.get("label_type")

                        if start is None or end is None:
                            continue

                        mention_text = text[start:end]

                        entities.append({
                            "text": mention_text,
                            "type": label
                        })

                    if entities:
                        yield {
                            "text": text,
                            "entities": entities
                        }

    def _bio_to_record(self, chars, labels):
        text = "".join(chars)
        entities = []

        i = 0
        while i < len(labels):
            label = labels[i]
            if label.startswith("B-"):
                ent_type = label[2:]
                start = i
                i += 1
                while i < len(labels) and labels[i].startswith("I-"):
                    i += 1
                ent_text = "".join(chars[start:i])
                entities.append({
                    "text": ent_text,
                    "type": ent_type.lower()
                })
            else:
                i += 1

        return {
            "text": text,
            "entities": entities
        }
