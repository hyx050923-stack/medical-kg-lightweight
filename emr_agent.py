import requests
import json
import joblib
import os
import sqlite3
from config import Config
from predict import predict_text  # 复用你写好的预测逻辑
from entity_alignment import MedicalEntityAligner
from entity_linker import UniversalEntityLinker

class MedicalGraphQuerier:
    """简单的图谱查询器"""
    def __init__(self, db_path):
        self.db_path = db_path

    def query_related_info(self, entity_name):
        """查询一个实体的所有邻居（双向查询：既查它导致的，也查治疗它的）"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        try:
            # 1. 先找实体ID
            cur.execute("SELECT id FROM entities WHERE name = ?", (entity_name,))
            row = cur.fetchone()
            if not row: return []
            entity_id = row['id']
            
            results = []
            
            # 2. 正向查询 (Source -> Target): 查症状、并发症
            # 例如: 高血压 -> has_symptom -> 头痛
            cur.execute("""
                SELECT DISTINCT r.relation_type, e.name as target_name, e.entity_type as target_type
                FROM relationships r
                JOIN entities e ON r.target_entity_id = e.id
                WHERE r.source_entity_id = ?
            """, (entity_id,))
            results.extend([dict(r) for r in cur.fetchall()])

            # 3. 【新增】反向查询 (Target <- Source): 查药物、手术
            # 例如: 阿司匹林(Source) -> treats -> 高血压(Target)
            cur.execute("""
                SELECT DISTINCT r.relation_type, e.name as source_name, e.entity_type as source_type
                FROM relationships r
                JOIN entities e ON r.source_entity_id = e.id
                WHERE r.target_entity_id = ? AND r.relation_type = 'treats'
            """, (entity_id,))
            
            # 把反向查到的结果也加进去，格式稍微变一下以示区别
            for r in cur.fetchall():
                results.append({
                    "relation_type": "treated_by",  # 语义转换: treats -> treated_by
                    "target_name": r['source_name'],
                    "target_type": r['source_type']
                })
                
            return results
            
        finally:
            conn.close()

class EMRAgent:
    def __init__(self):
        # 加载你的模型
        self.model_path = os.path.join(str(Config.MODEL_DIR), "aligner_rf.joblib")
        self.clf = joblib.load(self.model_path)
        self.aligner = MedicalEntityAligner()
        self.linker = UniversalEntityLinker(str(Config.DB_PATH))
        self.kg_querier = MedicalGraphQuerier(str(Config.DB_PATH))

    def generate_emr(self, patient_text):
        """生成电子病历 + 知识增强"""
        # 1. 实体识别与链接 (利用你之前的成果)
        prediction = predict_text(
            patient_text, self.clf, self.aligner, self.linker, topk=1, min_score=0.4
        )
        
        emr_data = {
            "original_text": patient_text,
            "findings": [],
            "knowledge_graph_enrichment": {} # 知识图谱补充的信息
        }

        # 2. 遍历识别出的实体，去图谱里查知识
        for item in prediction:
            mention = item['mention']
            linked_entity = item['linked_entity']
            
            record = {
                "mention": mention,
                "standard_name": linked_entity['entity_name'] if linked_entity else "NIL"
            }
            emr_data["findings"].append(record)
            
            # 如果成功链接到了标准实体，去KG查它的关联信息
            if linked_entity:
                std_name = linked_entity['entity_name']
                related_info = self.kg_querier.query_related_info(std_name)
                if related_info:
                    emr_data["knowledge_graph_enrichment"][std_name] = related_info

        return emr_data

    def chat_with_llm(self, patient_text, user_question):
        """
        核心功能：GraphRAG (图谱增强生成) + 本地 Ollama 调用
        """
        # 1. 生成结构化病历和知识
        context_data = self.generate_emr(patient_text)
        
        # 2. 构造 Prompt
        # DeepSeek-R1 这种推理模型喜欢清晰的指令，我们把 prompt 稍微优化一下
        prompt = f"""
【任务指令】
你是一个专业的医疗AI助手。请基于以下【病人自述】、【结构化病历】和【知识图谱医学事实】来回答用户的提问。

【病人自述】
{context_data['original_text']}

【提取的结构化病历】
{json.dumps(context_data['findings'], ensure_ascii=False, indent=2)}

【知识图谱关联知识 (医学事实)】
{json.dumps(context_data['knowledge_graph_enrichment'], ensure_ascii=False, indent=2)}

【用户问题】
{user_question}

【回答要求】
1. 必须结合图谱中提供的“医学事实”进行回答（例如具体的药物名称、关联症状）。
2. 语气专业、客观、亲切。
3. 如果图谱中没有相关信息，可以基于你的通用医学知识补充，但要说明。
"""
        
        print("="*40)
        print("⚡️ Prompt 已构建，正在请求本地 Ollama (deepseek-r1:1.5b)...")
        print("(DeepSeek 需要思考一下，请稍等...)")

        # 3. 调用本地 Ollama 接口
        try:
            url = "http://localhost:11434/api/chat"
            payload = {
                "model": "deepseek-r1:1.5b",  # ⚠️ 确保这里的名字和你 ollama list 里的一样
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": False,       # 关闭流式输出，一次性拿回结果（简单）
                "options": {
                    "temperature": 0.2 # 医疗场景温度低一点，让它严谨些
                }
            }
            
            # 发送请求
            response = requests.post(url, json=payload)
            response.raise_for_status() # 检查是否有网络错误
            
            # 解析结果
            result_json = response.json()
            answer = result_json.get("message", {}).get("content", "")
            
            # DeepSeek-R1 可能会输出 <think> 标签，那是它的思考过程，非常有意思，我们可以保留
            print("-" * 20 + " DeepSeek 回答 " + "-" * 20)
            print(answer)
            print("="*40)
            return answer

        except Exception as e:
            print(f"❌ 调用 Ollama 失败: {e}")
            print("请检查：\n1. Ollama 是否已启动？\n2. 模型名称 'deepseek-r1:1.5b' 是否正确？")
            return "系统暂时无法响应。"

# --- 运行演示 ---
if __name__ == "__main__":
    agent = EMRAgent()
    
    # 场景：病人描述
    text = "患者出现剧烈头痛，伴有发热症状，既往有高血压病史。"
    
    # 场景：医生/用户提问
    question = "根据患者的高血压病史，出现头痛有哪些风险？建议怎么治疗？"
    
    agent.chat_with_llm(text, question)