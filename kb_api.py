import numpy as np
from ollama import embeddings
import threading
from flask import Flask, request, jsonify
import os
import uuid
import hashlib

app = Flask(__name__)

class Kb:
    def __init__(self, content):
        # 直接使用传入的内容
        self.chunks = self.split_content(content)
        self.embeds = self.get_embeddings(self.chunks)

    # 拆分知识库
    @staticmethod
    def split_content(content):
        chunks = content.split('# ')
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        return chunks

    # 字符串转向量
    def get_embedding(self, chunk):
        res = embeddings(model='bge-m3', prompt=chunk)
        return res['embedding']

    def get_embeddings(self, chunks):
        embeds = []
        for chunk in chunks:
            embed = self.get_embedding(chunk)
            embeds.append(embed)
        return np.array(embeds)

    # 查询相似性向量
    def search(self, text):
        max_similarity = 0
        max_similarity_index = 0
        ask_embed = self.get_embedding(text)
        
        for kb_embed_index, kb_embed in enumerate(self.embeds):
            similarity = self.similarity(kb_embed, ask_embed)
            if similarity > max_similarity:
                max_similarity = similarity
                max_similarity_index = kb_embed_index
        
        return {
            "content": self.chunks[max_similarity_index],
            "similarity": float(max_similarity),
            "index": max_similarity_index
        }

    # 相似度计算
    @staticmethod
    def similarity(A, B):
        dot_product = np.dot(A, B)
        norm_A = np.linalg.norm(A)
        norm_B = np.linalg.norm(B)
        cosine_sim = dot_product / (norm_A * norm_B)
        return cosine_sim

class KbManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.knowledge_bases = {}
            return cls._instance
    
    def add_kb(self, tenant_id, content):
        """为租户添加知识库"""
        with self._lock:
            if tenant_id in self.knowledge_bases:
                raise ValueError(f"租户 {tenant_id} 的知识库已存在")
            self.knowledge_bases[tenant_id] = Kb(content)
    
    def get_kb(self, tenant_id):
        """获取租户的知识库"""
        with self._lock:
            kb = self.knowledge_bases.get(tenant_id)
            if kb is None:
                raise KeyError(f"租户 {tenant_id} 的知识库不存在")
            return kb
    
    def remove_kb(self, tenant_id):
        """移除租户的知识库"""
        with self._lock:
            if tenant_id not in self.knowledge_bases:
                raise KeyError(f"租户 {tenant_id} 的知识库不存在")
            del self.knowledge_bases[tenant_id]
    
    def search(self, tenant_id, text):
        """在指定租户的知识库中搜索"""
        kb = self.get_kb(tenant_id)
        return kb.search(text)

# 文件存储目录
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def generate_tenant_id():
    """生成唯一的租户ID"""
    return str(uuid.uuid4())

def save_file(file):
    """保存上传的文件并返回文件路径"""
    if file.filename == '':
        raise ValueError("未选择文件")
    
    # 生成唯一文件名
    filename = hashlib.md5(file.read()).hexdigest() + os.path.splitext(file.filename)[1]
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    
    # 重置文件指针并保存
    file.seek(0)
    file.save(file_path)
    
    return file_path

@app.route('/kb', methods=['POST'])
def create_knowledge_base():
    """创建知识库API"""
    try:
        # 获取租户ID或生成新ID
        tenant_id = request.form.get('tenant_id') or generate_tenant_id()
        
        if 'file' in request.files:
            # 处理文件上传
            file = request.files['file']
            file_path = save_file(file)
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        elif 'content' in request.form:
            # 直接使用文本内容
            content = request.form['content']
        else:
            return jsonify({
                "status": "error",
                "message": "必须提供文件或文本内容"
            }), 400
        
        # 创建知识库
        manager = KbManager()
        manager.add_kb(tenant_id, content)
        
        return jsonify({
            "status": "success",
            "tenant_id": tenant_id,
            "message": "知识库创建成功",
            "chunks_count": len(manager.get_kb(tenant_id).chunks)
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/kb/<tenant_id>', methods=['DELETE'])
def delete_knowledge_base(tenant_id):
    """删除知识库API"""
    try:
        manager = KbManager()
        manager.remove_kb(tenant_id)
        
        return jsonify({
            "status": "success",
            "message": f"租户 {tenant_id} 的知识库已删除"
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 404

@app.route('/kb/<tenant_id>/search', methods=['POST'])
def search_knowledge_base(tenant_id):
    """搜索知识库API"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                "status": "error",
                "message": "缺少查询参数"
            }), 400
        
        manager = KbManager()
        result = manager.search(tenant_id, data['query'])
        
        return jsonify({
            "status": "success",
            "tenant_id": tenant_id,
            "result": result
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/kb/<tenant_id>/info', methods=['GET'])
def get_kb_info(tenant_id):
    """获取知识库信息API"""
    try:
        manager = KbManager()
        kb = manager.get_kb(tenant_id)
        
        return jsonify({
            "status": "success",
            "tenant_id": tenant_id,
            "chunks_count": len(kb.chunks),
            "embeddings_shape": kb.embeds.shape
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
