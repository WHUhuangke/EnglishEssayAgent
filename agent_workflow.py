"""
智能体工作流模块
管理作文出题和批改的整个流程
集成LangChain的Agent和Chain
"""
from typing import Dict, Any, List, Optional, Tuple
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain import hub
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun

from knowledge_base import EssayKnowledgeBase
from essay_grader import EssayGrader
from config import (
    GRADE_NAME_MAPPING, 
    LEVEL_MAPPING, 
    ESSAY_GENRES,
    GRADE_LEVELS
)

class EssayAgentWorkflow:
    """作文智能体工作流"""
    
    def __init__(self, knowledge_base: EssayKnowledgeBase, essay_grader: EssayGrader):
        """
        初始化工作流
        
        Args:
            knowledge_base: 知识库实例
            essay_grader: 作文批改器实例
        """
        self.knowledge_base = knowledge_base
        self.essay_grader = essay_grader
        
        # 初始化工具
        self.tools = self._initialize_tools()
        
        # 初始化Agent
        self.agent = None
        self.agent_executor = None
        self._initialize_agent()
        
        # 初始化记忆
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    def _initialize_tools(self) -> List[BaseTool]:
        """初始化工具"""
        tools = []
        
        # 1. 题目搜索工具
        class PromptSearchTool(BaseTool):
            name: str = "prompt_search"
            description: str = "搜索作文题目，参数：grade(年级), level(水平), genre(文体), topic(主题)"
            
            def _run(self, grade: str, level: str, genre: str = None, topic: str = None, 
                    run_manager: CallbackManagerForToolRun = None) -> str:
                """运行搜索"""
                prompts = self.knowledge_base.search_prompts(
                    grade=grade,
                    level=level,
                    genre=genre,
                    topic=topic,
                    k=3
                )
                
                if not prompts:
                    return "未找到匹配的作文题目。"
                
                result = f"找到 {len(prompts)} 个作文题目:\n\n"
                for i, prompt in enumerate(prompts, 1):
                    result += f"{i}. {prompt.get('title', '未知题目')}\n"
                    result += f"   要求: {prompt.get('prompt', '')[:50]}...\n\n"
                
                return result
            
            async def _arun(self, grade: str, level: str, genre: str = None, topic: str = None, 
                          run_manager: CallbackManagerForToolRun = None) -> str:
                return self._run(grade, level, genre, topic, run_manager)
        
        # 为工具类设置知识库引用
        PromptSearchTool.knowledge_base = self.knowledge_base
        
        # 2. 语法检查工具
        class GrammarCheckTool(BaseTool):
            name: str = "grammar_check"
            description: str = "检查英语文本中的语法错误"
            
            def _run(self, text: str, run_manager: CallbackManagerForToolRun = None) -> str:
                """运行语法检查"""
                return self.essay_grader.check_grammar(text)
            
            async def _arun(self, text: str, run_manager: CallbackManagerForToolRun = None) -> str:
                return self._run(text, run_manager)
        
        GrammarCheckTool.essay_grader = self.essay_grader
        
        # 3. 词汇分析工具
        class VocabularyAnalysisTool(BaseTool):
            name: str = "vocabulary_analysis"
            description: str = "分析英语文本的词汇使用情况"
            
            def _run(self, text: str, run_manager: CallbackManagerForToolRun = None) -> str:
                """运行词汇分析"""
                return self.essay_grader.analyze_vocabulary(text)
            
            async def _arun(self, text: str, run_manager: CallbackManagerForToolRun = None) -> str:
                return self._run(text, run_manager)
        
        VocabularyAnalysisTool.essay_grader = self.essay_grader
        
        # 创建工具实例
        tools.append(PromptSearchTool())
        tools.append(GrammarCheckTool())
        tools.append(VocabularyAnalysisTool())
        
        return tools
    
    def _initialize_agent(self):
        """初始化LangChain Agent"""
        try:
            # 从LangChain Hub获取ReAct提示
            prompt = hub.pull("hwchase17/react")
            
            # 创建Agent
            self.agent = create_react_agent(
                llm=self.essay_grader.llm,
                tools=self.tools,
                prompt=prompt
            )
            
            # 创建Agent执行器
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                memory=self.memory,
                verbose=False,
                handle_parsing_errors=True,
                max_iterations=5
            )
            
            print("✓ 智能体初始化完成")
            
        except Exception as e:
            # print(f"⚠️  智能体初始化: {e}")
            print("")

            # print("  某些高级功能可能不可用")
    
    def search_essay_prompt(self, grade: str, level: str, 
                           genre: Optional[str] = None, 
                           topic: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        搜索作文题目
        
        Args:
            grade: 年级
            level: 英语水平
            genre: 文体类型（可选）
            topic: 主题（可选）
            
        Returns:
            作文题目信息，如果没有匹配则返回None
        """
        # 标准化输入
        normalized_grade = self._normalize_grade(grade)
        normalized_level = self._normalize_level(level)
        normalized_genre = self._normalize_genre(genre) if genre else None
        
        # 构建搜索查询
        query_parts = []
        if normalized_genre:
            query_parts.append(normalized_genre)
        if topic:
            query_parts.append(topic)
        
        search_query = " ".join(query_parts) if query_parts else "english essay"
        
        # 搜索知识库
        prompts = self.knowledge_base.search_prompts(
            grade=normalized_grade,
            level=normalized_level,
            genre=normalized_genre,
            topic=topic,
            k=5
        )
        
        if not prompts:
            # 如果没有找到，放宽条件搜索
            prompts = self.knowledge_base.search_prompts(
                grade=normalized_grade,
                level=normalized_level,
                genre=None,
                topic=None,
                k=3
            )
        
        if not prompts:
            # 如果还是没有找到，使用通用搜索
            prompts = self.knowledge_base.search_prompts(
                grade=None,
                level=None,
                genre=None,
                topic=None,
                k=3
            )
        
        # 选择最相关的一个
        if prompts:
            selected_prompt = prompts[0]
            
            # 确保返回的格式包含必要字段
            if "requirements" not in selected_prompt or not selected_prompt["requirements"]:
                selected_prompt["requirements"] = [
                    "请按照题目要求完成作文",
                    "注意语法和拼写",
                    "保持内容连贯"
                ]
            
            return selected_prompt
        
        return None
    
    def grade_essay(self, essay: str, prompt_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        批改作文
        
        Args:
            essay: 学生作文
            prompt_info: 作文题目信息
            
        Returns:
            批改结果
        """
        return self.essay_grader.grade_essay(essay, prompt_info)
    
    def run_agent_conversation(self, query: str) -> str:
        """
        运行Agent进行对话
        
        Args:
            query: 用户查询
            
        Returns:
            Agent的回复
        """
        if not self.agent_executor:
            return "智能体未初始化，无法进行对话。"
        
        try:
            result = self.agent_executor.invoke({"input": query})
            return result.get("output", "未收到回复。")
        except Exception as e:
            return f"对话出错: {str(e)}"
    
    def _normalize_grade(self, grade: str) -> str:
        """
        标准化年级标识
        
        Args:
            grade: 原始年级字符串
            
        Returns:
            标准化后的年级标识
        """
        # 尝试从映射中查找
        normalized = GRADE_NAME_MAPPING.get(grade)
        if normalized:
            return normalized
        
        # 检查是否已经是标准格式
        if grade in GRADE_LEVELS:
            return grade
        
        # 尝试模糊匹配
        grade_lower = grade.lower()
        for key, config in GRADE_LEVELS.items():
            if grade_lower in config.get("description", "").lower():
                return key
        
        # 默认值
        return "middle_school"
    
    def _normalize_level(self, level: str) -> str:
        """
        标准化水平标识
        
        Args:
            level: 原始水平字符串
            
        Returns:
            标准化后的水平标识
        """
        normalized = LEVEL_MAPPING.get(level.lower())
        if normalized:
            return normalized
        
        # 默认值
        return "intermediate"
    
    def _normalize_genre(self, genre: str) -> str:
        """
        标准化文体标识
        
        Args:
            genre: 原始文体字符串
            
        Returns:
            标准化后的文体标识
        """
        # 检查是否已经是英文标识
        if genre in ESSAY_GENRES:
            return genre
        
        # 检查是否是中文
        for eng, chi in ESSAY_GENRES.items():
            if genre == chi:
                return eng
        
        # 尝试模糊匹配
        genre_lower = genre.lower()
        for eng, chi in ESSAY_GENRES.items():
            if genre_lower in eng.lower() or genre_lower in chi.lower():
                return eng
        
        # 返回原始值
        return genre
    
    def get_learning_recommendations(self, grade: str, level: str, 
                                   recent_topics: List[str] = None) -> List[Dict[str, Any]]:
        """
        获取学习推荐
        
        Args:
            grade: 年级
            level: 水平
            recent_topics: 最近练习的主题
            
        Returns:
            推荐题目列表
        """
        normalized_grade = self._normalize_grade(grade)
        normalized_level = self._normalize_level(level)
        
        # 获取所有题目
        all_prompts = self.knowledge_base.get_all_prompts()
        
        # 过滤和排序
        recommendations = []
        for prompt in all_prompts:
            # 匹配年级和水平
            if (prompt.get('grade') == normalized_grade and 
                prompt.get('level') == normalized_level):
                
                # 避免最近练习过的主题
                if recent_topics and prompt.get('topic') in recent_topics:
                    continue
                    
                recommendations.append(prompt)
                
                if len(recommendations) >= 5:  # 最多5个推荐
                    break
        
        # 如果不够，放宽年级要求
        if len(recommendations) < 3:
            for prompt in all_prompts:
                if prompt.get('level') == normalized_level and prompt not in recommendations:
                    recommendations.append(prompt)
                    
                if len(recommendations) >= 5:
                    break
        
        return recommendations
    
    def analyze_learning_progress(self, grading_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析学习进度
        
        Args:
            grading_history: 批改历史记录
            
        Returns:
            进度分析结果
        """
        if not grading_history:
            return {
                "total_essays": 0,
                "average_score": 0,
                "improvement": 0,
                "common_errors": [],
                "strengths": [],
                "weaknesses": [],
                "suggestions": []
            }

