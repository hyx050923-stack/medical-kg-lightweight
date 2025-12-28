# medical-kg-lightweight

轻量级中文医疗知识图谱构建工程（规则 + 传统 ML）。  
设计目标：无深度学习、无重型依赖，适配中小规模医疗文本的快速落地需求。  
主要技术栈：jieba（分词） + fuzzywuzzy/rapidfuzz（模糊匹配） + scikit-learn（传统 ML） + SQLite（轻量化存储）。

核心思想（简洁）
- 文本预处理 → 规则 + 词典抽取实体 → 模糊匹配 + 特征化 + 传统 ML 完成实体对齐/消歧 → 规则优先 + ML 兜底关系抽取 → 入库 SQLite 形成知识图谱。

链接
- Yidu-S4K 数据集（建议用于阶段一训练/初始化实体库）：https://tianchi.aliyun.com/dataset/144419
- Yidu-S4K 说明（备用）：http://data.openkg.cn/dataset/groups/yidu-s4k

---

## 目录结构（示例）
- config.py                 全局配置
- logger_config.py          日志配置
- text_preprocessor.py      文本清洗与分词（jieba）
- data_loader.py            Yidu-S4K 数据加载器
- entity_recognizer.py      规则 + 词典的实体识别
- entity_linker.py          通用实体链接（基于 fuzzy/rapidfuzz）
- entity_alignment.py       实体对齐特征与简单判定器
- kg_builder.py             SQLite KG 构建（批量/事务/单例连接）
- run.py             演示脚本：从 Yidu-S4K 提取实体并构建 KG、识别+链接示例
- requirements.txt          依赖（见说明）

---

## 快速开始（5 分钟跑通阶段一）

1. 克隆仓库（或把代码放到项目根）
   ```bash
   git clone https://github.com/hyx050923-stack/medical-kg-lightweight.git
   cd medical-kg-lightweight
   ```

2. 建议使用虚拟环境
   - Linux / macOS:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```

3. 安装依赖
   注意：Python 自带 sqlite3，无需另外安装 `sqlite3-python`。如果 requirements.txt 中包含该行，请先删除或注释。
   ```bash
   pip install -r requirements.txt
   # 推荐替换 fuzzywuzzy -> rapidfuzz（速度更快），若出现 fuzzywuzzy 安装问题可：
   pip install rapidfuzz
   pip install python-Levenshtein   # 可选，加速 fuzzywuzzy（若保留 fuzzywuzzy）
   ```

4. 下载并准备 Yidu-S4K 数据
   - 将 Yidu-S4K 数据解压到项目目录的 `data/yidu_s4k/`，或任意路径并设置环境变量：
     - Linux / macOS:
       ```bash
       export YIDU_S4K_PATH=/full/path/to/data/yidu_s4k
       ```
     - Windows:
       ```powershell
       setx YIDU_S4K_PATH "C:\full\path\to\data\yidu_s4k"
       ```
   - 确认文件名（示例）：`subtask1_training_part1.txt`、`subtask1_training_part2.txt`、`subtask1_test_set_with_answer.json` 等。

5. 可选：准备 jieba 医疗词典与停用词
   - 放 `medical_terms.txt` 和 `stopwords.txt` 到 `data/`（config 中有默认路径）。

6. 运行阶段一脚本（示例）
   ```bash
   python run_stage1.py
   ```
   脚本流程：
   - 初始化日志与目录
   - 加载 Yidu-S4K 并提取实体（可通过参数 limit 控制用于测试的数据量）
   - 批量写入 SQLite（默认 db：`medical_kg.db`）
   - 使用示例文本进行实体识别并尝试链接到实体库

7. 查看结果
   - 日志：`./logs/kg_builder.log`、`./logs/errors.log`
   - SQLite 数据库：`./medical_kg.db`（可用 DB Browser for SQLite 或 sqlite3 CLI 检查 `entities` / `aliases` / `relationships` 表）

---

## 常见配置（config.py）
- 数据路径：`Config.YIDU_S4K_DIR` 或环境变量 `YIDU_S4K_PATH`
- 数据库：`Config.DB_PATH`
- 实体链接阈值：`Config.ENTITY_LINKING_THRESHOLD`
- 实体对齐阈值：`Config.ALIGNMENT_THRESHOLD`
- 若要把模糊匹配改为 rapidfuzz，请在 `entity_linker.py` 中把 fuzzywuzzy 的 `fuzz` 替换为 `rapidfuzz.fuzz`，接口兼容。

---

## 性能建议（重要）
- 当实体库 > 1k 时，避免全库两两比较：
  - 实现 blocking（按实体类型、长度差、关键词过滤候选）能削减 90%+ 无效候选；
  - 使用 rapidfuzz 替代 fuzzywuzzy 可显著提速；
  - 预加载实体缓存（实现已包含）并避免频繁 DB 查询。
- 批量写入时使用事务（kg_builder 已内置批量接口与事务上下文）。

---

## 故障排查
- 找不到数据或文件路径错误：确认 `YIDU_S4K_PATH` 指向包含 `subtask1_training_part*.txt` 的目录。
- sqlite 锁或并发写入错误：请确保单进程写入或使用 kg_builder 中的单例连接；避免多进程同时写同一文件。
- fuzzywuzzy 过慢或安装失败：优先安装 `python-Levenshtein` 或直接使用 `rapidfuzz`。
- jieba 未识别医学词：加载自定义词典 `data/medical_terms.txt` 并在 TextPreprocessor 中指定路径。

---

## 下一步建议（可选）
- 为 entity_linker 添加 blocking 模块（按 type/length/keywords 过滤候选）。
- 使用 `entity_alignment.extract_features` 构建训练集并用随机森林训练实体对齐分类器。
- 关系抽取从二分类扩展为多分类（不同的关系类型）。
- 增强异常处理与单元测试，完善 CI。

---

## 贡献 & 许可证
- 欢迎 Issues / PR（请在提交前确保不包含敏感病历原文）。  
- 本项目未包含商业授权声明；请根据你的组织政策与 Yidu-S4K 使用许可处理数据与分发。

---

如需我为你：
- 生成带有 blocking 的 entity_linker 示例；
- 把 fuzzywuzzy 全部替换为 rapidfuzz 的 PR 补丁；
- 或者自动生成更详细的部署文档与 demo 数据集子集（用于快速测试）——告诉我你想做哪一项，我会直接把修改/补丁内容给你。