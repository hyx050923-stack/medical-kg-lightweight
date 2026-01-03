import logging
from tqdm import tqdm
from config import Config
from data_loader import YiduS4KDataLoader
from kg_builder import MedicalKGBuilder
from logger_config import setup_logging
from entity_recognizer import MedicalEntityRecognizer # 复用你的识别器

def build_graph():
    setup_logging()
    
    # 1. 初始化
    db_path = str(Config.DB_PATH)
    builder = MedicalKGBuilder(db_path)
    loader = YiduS4KDataLoader(Config.YIDU_S4K_DIR)
    recognizer = MedicalEntityRecognizer() # 实例化识别器
    
    # =======================================================
    # 阶段一：从 Task 2 Excel 读取高精度的“属性关系”
    # =======================================================
    print("\n[阶段一] 正在读取 Excel 构建肿瘤属性关系...")
    count_excel = 0
    for record in loader.load_task2_training():
        for rel in record.get('relations', []):
            subj = rel['subject']
            pred = rel['predicate']
            obj = rel['object']
            
            # 存入图谱
            s_id = builder.add_entity(subj, "Disease") # 假设主体是疾病
            o_id = builder.add_entity(obj, "Attribute")
            if s_id and o_id:
                builder.add_relationship(s_id, pred, o_id, confidence=0.95)
                count_excel += 1
    print(f"✅ Excel 数据导入完成，共 {count_excel} 条。")

    # =======================================================
    # 阶段二：从 Task 1 文本挖掘“药物/治疗”通用关系 (新增功能！)
    # =======================================================
    print("\n[阶段二] 正在分析文本，挖掘 药物-疾病、治疗-疾病 关系...")
    count_mining = 0
    
    # 遍历 Task 1 的训练数据 (全是纯文本)
    # limit=200 防止跑太久，大作业演示足够了
    for record in tqdm(loader.load_task1_training(split='all'), desc="挖掘中"):
        text = record.get('originalText') or record.get('text')
        if not text: continue
        
        # 1. 用你的识别器提取实体
        # 结果格式: [(name, type, span), ...]
        entities = recognizer.recognize(text)
        
        # 2. 对实体进行分类
        diseases = [e[0] for e in entities if e[1] == 'disease']
        drugs = [e[0] for e in entities if e[1] == 'drug']
        treatments = [e[0] for e in entities if e[1] == 'treatment']
        symptoms = [e[0] for e in entities if e[1] == 'symptom']
        
        # 3. 应用“共现规则”构建关系
        # 规则 A: 药物 -> 治疗 -> 疾病
        for drug in drugs:
            for disease in diseases:
                s_id = builder.add_entity(drug, "drug")
                o_id = builder.add_entity(disease, "disease")
                # 存入关系: confidence 给低一点(0.6)，因为是猜的
                builder.add_relationship(s_id, "treats", o_id, confidence=0.6)
                count_mining += 1
                
        # 规则 B: 手术/操作 -> 治疗 -> 疾病
        for treat in treatments:
            for disease in diseases:
                s_id = builder.add_entity(treat, "treatment")
                o_id = builder.add_entity(disease, "disease")
                builder.add_relationship(s_id, "treats", o_id, confidence=0.6)
                count_mining += 1
                
        # 规则 C: 疾病 -> 伴随 -> 症状
        for disease in diseases:
            for symptom in symptoms:
                s_id = builder.add_entity(disease, "disease")
                o_id = builder.add_entity(symptom, "symptom")
                builder.add_relationship(s_id, "has_symptom", o_id, confidence=0.7)
                count_mining += 1

    print(f"✅ 文本挖掘完成！共自动发现 {count_mining} 条通用关系。")
    print(f"🎉 知识图谱构建完毕！总计关系: {count_excel + count_mining}")

if __name__ == "__main__":
    build_graph()