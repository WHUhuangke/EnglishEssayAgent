"""
知识库模块
管理作文题目的向量存储和检索
使用LangChain的Chroma向量数据库
"""
import os
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever

from config import (
    VECTOR_STORE_CONFIG,
    SAMPLE_ESSAY_PROMPTS,
    GRADE_LEVELS,
    ESSAY_GENRES
)

class EssayKnowledgeBase:
    """作文题目向量知识库"""
    
    def __init__(self, persist_directory: str = None, collection_name: str = None):
        """
        初始化知识库
        
        Args:
            persist_directory: 向量数据库持久化目录
            collection_name: 集合名称
        """
        if persist_directory is None:
            persist_directory = VECTOR_STORE_CONFIG["persist_directory"]
        if collection_name is None:
            collection_name = VECTOR_STORE_CONFIG["collection_name"]
        
        # 确保目录存在
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # 初始化嵌入模型
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
        
        # 初始化向量存储
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
        
        # 初始化检索器
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 5}
        )
        
        # 如果知识库为空，初始化示例数据
        if self.is_empty():
            self._initialize_sample_data()
    
    def is_empty(self) -> bool:
        """
        检查知识库是否为空
        
        Returns:
            是否为空
        """
        try:
            count = self.vector_store._collection.count()
            return count == 0
        except:
            return True
    
    def _initialize_sample_data(self):
        """初始化示例数据"""
        print("正在初始化示例作文题目...")
        
        try:
            # 从JSON文件读取数据
            with open('essays.json', 'r', encoding='utf-8') as f:
                essay_prompts = json.load(f)
                
            for prompt_data in essay_prompts:
                self.add_essay_prompt(prompt_data)
            
            print(f"✓ 已从文件添加 {len(essay_prompts)} 个示例作文题目")
        except FileNotFoundError:
            print("错误：未找到essays.json文件")
        except json.JSONDecodeError as e:
            print(f"错误：JSON文件解析失败 - {e}")
        
    def add_essay_prompt(self, prompt_data: Dict[str, Any]) -> str:
        """
                添加作文题目到知识库
        
        Args:
            prompt_data: 作文题目数据
            
        Returns:
            文档ID
        """
        # 创建文档内容
        content_parts = [
            f"Title: {prompt_data.get('title', 'Untitled')}",
            f"Prompt: {prompt_data.get('prompt', '')}",
            f"Grade: {prompt_data.get('grade', '')}",
            f"Level: {prompt_data.get('level', '')}",
            f"Genre: {prompt_data.get('genre', '')}",
            f"Topic: {prompt_data.get('topic', '')}",
        ]
        
        # 添加要求
        requirements = prompt_data.get('requirements', [])
        if requirements:
            # print('requirements')
            content_parts.append("Requirements:")
            for req in requirements:
                content_parts.append(f"- {req}")
        
        # 添加关键词
        keywords = prompt_data.get('keywords', [])
        if keywords:
            content_parts.append(f"Keywords: {', '.join(keywords)}")
        
        content = "\n".join(content_parts)
        
        # 创建元数据
        metadata = {
            "grade": prompt_data.get('grade', ''),
            "level": prompt_data.get('level', ''),
            "genre": prompt_data.get('genre', ''),
            "topic": prompt_data.get('topic', ''),
            "title": prompt_data.get('title', ''),
            "prompt_id": str(prompt_data.get('id', 'unknown'))
        }
        
        # 创建文档
        doc = Document(
            page_content=content,
            metadata=metadata
        )
        
        # 添加到向量存储
        self.vector_store.add_documents([doc])
        
        return f"Added prompt: {prompt_data.get('title', 'Untitled')}"

    def search_by_semantic_similarity(self, query: str, k: int = 5, 
                                     filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        基于语义相似度搜索作文题目
        
        Args:
            query: 搜索查询
            k: 返回结果数量
            filters: 过滤条件
            
        Returns:
            匹配的作文题目列表
        """
        # 执行语义搜索
        docs = self.vector_store.similarity_search(query, k=k)
        # print(f"Found {len(docs)} documents for query: '{query}'")
        # print('docs', docs)
        # 应用过滤条件
        filtered_results = []
        for doc in docs:
            metadata = doc.metadata
            
            # 应用过滤条件
            if filters:
                should_include = True
                for key, value in filters.items():
                    if metadata.get(key) != value:
                        should_include = False
                        break
                
                if not should_include:
                    continue
            
            # 解析文档
            prompt_data = self._parse_document_to_prompt(doc)
            filtered_results.append(prompt_data)
        
        return filtered_results
    
    def search_prompts(self, grade: str = None, level: str = None, 
                      genre: str = None, topic: str = None, k: int = 5) -> List[Dict[str, Any]]:
        """
        搜索作文题目
        
        Args:
            grade: 年级
            level: 水平
            genre: 文体
            topic: 主题
            k: 返回结果数量
            
        Returns:
            作文题目列表
        """
        # 构建搜索查询
        query_parts = []
        if genre:
            query_parts.append(f"{genre} essay")
        if topic:
            query_parts.append(topic)
        
        search_query = " ".join(query_parts) if query_parts else "english essay"
        
        # 构建过滤条件
        filters = {}
        if grade:
            filters['grade'] = grade
        if level:
            filters['level'] = level
        if genre:
            filters['genre'] = genre
        if topic:
            filters['topic'] = topic
        # print('search_query:',search_query)
        # print('filters:',filters)
        # 执行搜索
        return self.search_by_semantic_similarity(
            query=search_query,
            k=k,
            filters=filters if filters else None
        )
    
    def _parse_document_to_prompt(self, doc: Document) -> Dict[str, Any]:
        """
        将文档解析为作文题目数据
        
        Args:
            doc: LangChain文档
            
        Returns:
            作文题目数据字典
        """
        content = doc.page_content
        metadata = doc.metadata
        
        # 解析内容
        lines = content.split('\n')
        
        prompt_data = {
            "title": metadata.get('title', ''),
            "prompt": "",
            "grade": metadata.get('grade', ''),
            "level": metadata.get('level', ''),
            "genre": metadata.get('genre', ''),
            "topic": metadata.get('topic', ''),
            "requirements": [],
            "full_content": content
        }
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith("Title:"):
                prompt_data["title"] = line.replace("Title:", "").strip()
            elif line.startswith("Prompt:"):
                prompt_data["prompt"] = line.replace("Prompt:", "").strip()
            elif line.startswith("Requirements:"):
                current_section = "requirements"
            elif line.startswith("Keywords:"):
                current_section = None
            elif current_section == "requirements" and line.startswith("-"):
                req = line[1:].strip()
                if req:
                    prompt_data["requirements"].append(req)
        
        return prompt_data
    
    def get_all_prompts(self) -> List[Dict[str, Any]]:
        """
        获取所有作文题目
        
        Returns:
            所有作文题目列表
        """
        # 获取所有文档
        docs = self.vector_store.get()
        
        prompts = []
        if docs and 'documents' in docs:
            for i, doc_content in enumerate(docs['documents']):
                metadata = docs['metadatas'][i] if docs['metadatas'] else {}
                
                # 创建文档对象
                doc = Document(
                    page_content=doc_content,
                    metadata=metadata
                )
                
                # 解析为prompt数据
                prompt_data = self._parse_document_to_prompt(doc)
                prompts.append(prompt_data)
        
        return prompts
    
    def get_prompt_count(self) -> int:
        """
        获取作文题目数量
        
        Returns:
            题目数量
        """
        try:
            return self.vector_store._collection.count()
        except:
            return 0
    
    def clear(self):
        """清空知识库"""
        self.vector_store.delete_collection()
        print("知识库已清空")
    
    def save_to_json(self, filepath: str = "data/essay_prompts_backup.json"):
        """
        将知识库保存到JSON文件
        
        Args:
            filepath: 文件路径
        """
        prompts = self.get_all_prompts()
        
        # 确保目录存在
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, ensure_ascii=False, indent=2)
        
        print(f"知识库已备份到: {filepath}")
    
    def load_from_json(self, filepath: str = "data/essay_prompts_backup.json"):
        """
        从JSON文件加载知识库
        
        Args:
            filepath: 文件路径
        """
        if not Path(filepath).exists():
            print(f"文件不存在: {filepath}")
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        
        for prompt_data in prompts:
            self.add_essay_prompt(prompt_data)
        
        print(f"从 {filepath} 加载了 {len(prompts)} 个作文题目")

# 测试函数
def test_knowledge_base():
    """测试知识库功能"""
    print("测试知识库模块...")
    
    # 创建知识库实例
    kb = EssayKnowledgeBase()
    
    # 测试添加文档
    test_prompt = {
        "id": 999,
        "title": "Test Essay Prompt",
        "prompt": "This is a test prompt for testing the knowledge base functionality.",
        "grade": "middle_school_2",
        "level": "intermediate",
        "genre": "test",
        "topic": "test",
        "requirements": ["Test requirement 1", "Test requirement 2"],
        "keywords": ["test", "example", "demo"]
    }
    
    result = kb.add_essay_prompt(test_prompt)
    print(f"添加测试文档: {result}")
    
    # 测试搜索
    print("\n测试语义搜索...")
    results = kb.search_by_semantic_similarity("test essay", k=3)
    print(f"搜索到 {len(results)} 个结果")
    
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result.get('title', 'Unknown')}")
    
    # 测试带过滤的搜索
    print("\n测试带过滤的搜索...")
    filtered_results = kb.search_prompts(
        grade="middle_school_2",
        level="intermediate",
        genre="test",
        k=3
    )
    print(f"过滤搜索到 {len(filtered_results)} 个结果")
    
    # 测试获取所有文档
    all_prompts = kb.get_all_prompts()
    print(f"\n知识库中共有 {len(all_prompts)} 个文档")
    
    # 测试计数
    count = kb.get_prompt_count()
    print(f"知识库文档数量: {count}")
    
    # 清理测试数据
    print("\n清理测试数据...")
    kb.clear()
    
    # 重新初始化示例数据
    kb._initialize_sample_data()
    print("重新初始化示例数据完成")
    
    # 测试备份
    print("\n测试备份功能...")
    kb.save_to_json("data/test_backup.json")
    
    print("\n✅ 知识库测试完成")

if __name__ == "__main__":
    # 检查环境变量
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("请设置 OPENAI_API_KEY 环境变量")
    else:
        test_knowledge_base()

