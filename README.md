# 🏥 Medical-KG-Lightweight: 基于知识图谱的医疗 GraphRAG 系统

> **课程大作业级别项目** | **轻量级** | **完全本地化** | **可解释性**

本项目实现了一个端到端的**医疗知识图谱构建与应用系统**。它能够从非结构化的电子病历（EMR）中提取实体与关系，构建本地知识图谱（SQLite），并结合大语言模型（DeepSeek-R1 via Ollama）实现**检索增强生成（GraphRAG）**，为用户提供基于医学事实的智能问答与病历结构化服务。

---

## ✨ 核心功能 (Key Features)

### 1. 🧠 自动化知识图谱构建 (Auto-KG Construction)

* **多源融合**：支持从结构化数据（Excel/CSV）导入精准属性，同时从非结构化文本中挖掘潜在的共现关系（如：药物-治疗-疾病）。
* **智能去重**：内置数据库约束与逻辑检查，防止重复知识入库。
* **别名注入**：支持医疗术语归一化（如 `HTN` -> `高血压`，`MI` -> `心肌梗死`）。

### 2. 🔗 高精度实体链接 (Medical Entity Linking)

* **混合模型**：结合了 **规则匹配**、**模糊检索 (RapidFuzz)** 和 **机器学习排序 (Random Forest)**。
* **鲁棒性**：能够处理医疗缩写、错别字及长短词匹配（如 `心肌梗死` ↔ `急性下壁心肌梗死`）。
* **拒识机制**：有效过滤“切除”、“检查”等无意义的泛义词。

### 3. 🤖 GraphRAG 智能诊断助手 (AI Doctor Agent)

* **病历结构化**：自动提取病历中的关键实体，生成标准化 JSON 报告。
* **图谱增强 (RAG)**：在回答用户问题时，自动检索图谱中的关联知识（如并发症、禁忌症、推荐药物）。
* **本地大模型**：通过 Ollama 集成 `deepseek-r1:1.5b`，实现完全离线的隐私保护问答。

---

## 🛠️ 技术栈 (Tech Stack)

* **语言**: Python 3.9+
* **知识库**: SQLite (轻量级图谱存储)
* **NLP & ML**: `scikit-learn`, `rapidfuzz`, `jieba`, `numpy`, `pandas`
* **LLM 框架**: Ollama (DeepSeek-R1), `requests`
* **数据集支持**: Yidu-S4K (CCKS 2019)

---

## 🚀 快速开始 (Quick Start)

### 第一步：环境准备

1. **安装 Python 依赖**：
```bash
pip install -r requirements.txt

```


*(注：需包含 pandas, openpyxl, scikit-learn, joblib, rapidfuzz, tqdm, requests, jieba)*
2. **安装并启动 Ollama**（用于大模型支持）：
* 下载安装 [Ollama](https://ollama.com/)。
* 拉取 DeepSeek 模型：
```bash
ollama pull deepseek-r1:1.5b

```


* 保持 Ollama 在后台运行。



### 第二步：准备数据

将 **Yidu-S4K** 数据集文件放入 `data/yidu_s4k/` 目录：

* `subtask1_training_part1.txt` (用于挖掘关系)
* `subtask2_training_part1.xlsx` (用于导入属性)

### 第三步：构建知识图谱

运行构建脚本，系统将自动进行数据清洗、实体提取、关系挖掘并存入 `medical_kg.db`。

```bash
python build_full_kg.py

```

*可选：注入常见医疗别名（增强实体链接能力）*

```bash
python inject_aliases.py

```

### 第四步：训练排序模型 (可选)

如果你需要重新训练实体对齐模型：

```bash
python train.py

```

### 第五步：启动智能医疗助手

运行 Agent，体验基于图谱的医疗问答：

```bash
python emr_agent.py

```

---

## 📂 项目结构

```text
medical-kg-lightweight/
├── data/                   # 数据存放目录
│   └── yidu_s4k/           # Yidu-S4K 数据集
├── models/                 # 模型保存目录 (Random Forest)
├── logs/                   # 运行日志
├── medical_kg.db           # SQLite 知识图谱文件
│
├── kg_builder.py           # 数据库底层操作 (DAO层)
├── build_full_kg.py        # 图谱构建脚本 (ETL + 文本挖掘)
├── inject_aliases.py       # 别名注入工具
│
├── entity_recognizer.py    # 命名实体识别 (NER)
├── entity_linker.py        # 实体召回 (Blocking)
├── entity_alignment.py     # 实体对齐特征提取
├── train.py                # 排序模型训练脚本
├── predict.py              # 实体链接推理接口
│
├── emr_agent.py            # [核心] 智能病历 Agent + GraphRAG
├── data_loader.py          # 数据加载器 (支持 JSONL/Excel)
├── config.py               # 全局配置
└── requirements.txt        # 项目依赖

```

---

## 📊 效果展示

### 场景：高血压患者咨询

**输入病历**：

> "患者出现剧烈头痛，伴有发热症状，既往有高血压病史。"

**系统处理流程**：

1. **实体提取**：`头痛`、`发热`、`高血压`
2. **知识检索**：
* `高血压` --(has_symptom)--> `头晕`, `头痛`, `心悸`
* `阿司匹林` --(treated_by)--> `高血压`


3. **LLM 生成**：
> "根据患者的高血压病史...图谱显示高血压常伴随头痛风险...建议监测血压，并在医生指导下使用阿司匹林或降压药物..."



---

## 📝 引用与致谢

* 数据集来源：CCKS 2019 Yidu-S4K
* 模型支持：DeepSeek-AI

---

**© 2026 Medical-KG-Lightweight Course Project**