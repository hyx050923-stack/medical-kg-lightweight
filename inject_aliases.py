import sqlite3
from config import Config

def inject_medical_aliases():
    print(f"正在连接数据库: {Config.DB_PATH}")
    conn = sqlite3.connect(str(Config.DB_PATH))
    cur = conn.cursor()
    
    # 定义需要注入的知识 (标准名 -> 别名列表)
    knowledge = {
        "高血压": ["HTN", "高血压病"],
        "冠心病": ["CAD", "冠状动脉粥样硬化性心脏病"],
        "心肌梗死": ["MI", "急性心肌梗死"], # 如果库里叫心肌梗死
        "急性心肌梗死": ["MI", "心肌梗死"], # 如果库里叫急性心肌梗死
        "糖尿病": ["DM", "消渴症"],
        "慢性阻塞性肺病": ["COPD", "慢阻肺"]
    }
    
    for name, aliases in knowledge.items():
        # 1. 先找实体ID
        cur.execute("SELECT id FROM entities WHERE name = ?", (name,))
        row = cur.fetchone()
        
        if row:
            eid = row[0]
            for alias in aliases:
                try:
                    # 2. 插入别名
                    cur.execute("INSERT INTO aliases (entity_id, alias) VALUES (?, ?)", (eid, alias))
                    print(f"✅ 成功注入: {name} <- {alias}")
                except sqlite3.IntegrityError:
                    print(f"⚠️ 已存在: {name} <- {alias}")
        else:
            print(f"❌ 未找到实体: {name} (请检查数据库中该实体的实际名称)")

    conn.commit()
    conn.close()
    print("注入完成！")

if __name__ == "__main__":
    inject_medical_aliases()