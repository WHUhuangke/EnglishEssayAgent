"""
作文批改模块
使用大模型和工具对作文进行评分和批改
集成LangChain的LLM和自定义工具
"""
import os
import re
from typing import Dict, Any, List, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from config import MODEL_CONFIG, GRADING_CONFIG

class EssayGrader:
    """作文批改器"""
    
    def __init__(self, model_name: str = None, temperature: float = None):
        """
        初始化批改器
        
        Args:
            model_name: 模型名称
            temperature: 温度参数
        """
        if model_name is None:
            model_name = MODEL_CONFIG.get("default_model", "gpt-3.5-turbo")
        if temperature is None:
            temperature = MODEL_CONFIG.get("temperature", 0.3)
        
        # 初始化大模型
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # 初始化语法检查工具
        self.grammar_tool = None
        self._initialize_grammar_tool()
        
        # 初始化批改Chain
        self._initialize_grading_chain()
    
    def _initialize_grammar_tool(self):
        """初始化语法检查工具"""
        try:
            import language_tool_python
            self.grammar_tool = language_tool_python.LanguageTool('en-US')
            print("✓ 语法检查工具初始化成功")
        except ImportError:
            print("⚠️  未安装language-tool-python，语法检查功能受限")
            print("  安装: pip install language-tool-python")
        except Exception as e:
            print(f"⚠️  语法检查工具初始化失败: {e}")
    
    def _initialize_grading_chain(self):
        """初始化批改Chain"""
        # 作文批改提示词模板
        grading_template = """你是一位经验丰富的英语老师，负责批改学生的英语作文。

作文题目: {title}
写作要求: {requirements}
学生作文: {essay}

请从以下方面进行批改，并给出具体的分数和反馈：

1. 语法准确性 (0-30分)
   - 检查时态、语态、主谓一致等基本语法
   - 检查句子结构是否完整
   - 检查标点符号使用是否正确

2. 词汇使用 (0-30分)
   - 检查词汇是否准确恰当
   - 评估词汇的丰富性和多样性
   - 检查拼写错误

3. 内容完整性 (0-40分)
   - 是否紧扣题目要求
   - 内容是否充实、有逻辑
   - 结构是否清晰，是否有开头、主体、结尾

请按以下格式回复：

总体评分: [分数]/100
语法得分: [分数]/30
词汇得分: [分数]/30
内容得分: [分数]/40

总体评价: [这里写你的总体评价，至少3句话]

具体反馈:
1. 优点: [列出2-3个优点]
2. 需要改进的地方: [列出2-3个需要改进的地方]
3. 具体建议: [给出具体的改进建议]"""

        self.grading_prompt = PromptTemplate(
            input_variables=["title", "requirements", "essay"],
            template=grading_template
        )
        
        # 创建LLMChain
        self.grading_chain = LLMChain(
            llm=self.llm,
            prompt=self.grading_prompt,
            verbose=False
        )
    
    def grade_essay(self, essay: str, prompt_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        批改作文
        
        Args:
            essay: 学生作文
            prompt_info: 作文题目信息
            
        Returns:
            批改结果字典
        """
        # 1. 基础检查
        if not essay or len(essay.strip()) < 10:
            return self._get_empty_essay_result()
        
        # 2. 使用大模型进行综合批改
        llm_result = self._grade_with_llm(essay, prompt_info)
        
        # 3. 语法检查
        grammar_results = self.check_grammar(essay, return_dict=True)
        
        # 4. 词汇分析
        vocabulary_results = self.analyze_vocabulary(essay, return_dict=True)
        
        # 5. 计算最终分数
        final_scores = self._calculate_final_scores(llm_result, grammar_results, vocabulary_results)
        
        # 6. 整合结果
        result = {
            "essay": essay,
            "prompt_title": prompt_info.get("title", "Unknown"),
            "overall_score": final_scores["overall"],
            "grammar_score": final_scores["grammar"],
            "vocabulary_score": final_scores["vocabulary"],
            "content_score": final_scores["content"],
            "overall_feedback": llm_result.get("overall_feedback", ""),
            "grammar_errors": grammar_results.get("errors", []),
            "vocabulary_feedback": vocabulary_results.get("feedback", ""),
            "suggestions": llm_result.get("suggestions", []),
            "strengths": llm_result.get("strengths", []),
            "weaknesses": llm_result.get("weaknesses", []),
            "word_count": len(essay.split()),
            "character_count": len(essay)
        }
        
        return result
    
    def _grade_with_llm(self, essay: str, prompt_info: Dict[str, Any]) -> Dict[str, Any]:
        """使用大模型进行批改"""
        try:
            # 调用LLMChain
            response = self.grading_chain.run(
                title=prompt_info.get("title", "Essay"),
                requirements=prompt_info.get("prompt", ""),
                essay=essay
            )
            
            # 解析响应
            return self._parse_llm_response(response)
            
        except Exception as e:
            print(f"大模型批改出错: {e}")
            return self._get_default_grading_result()
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """解析大模型的响应"""
        result = {
            "overall_score": 0,
            "grammar_score": 0,
            "vocabulary_score": 0,
            "content_score": 0,
            "overall_feedback": "",
            "strengths": [],
            "weaknesses": [],
            "suggestions": []
        }
        
        try:
            lines = response_text.strip().split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                
                # 提取分数
                if line.startswith("总体评分:"):
                    match = re.search(r'(\d+)/100', line)
                    if match:
                        result["overall_score"] = int(match.group(1))
                
                elif line.startswith("语法得分:"):
                    match = re.search(r'(\d+)/30', line)
                    if match:
                        result["grammar_score"] = int(match.group(1))
                
                elif line.startswith("词汇得分:"):
                    match = re.search(r'(\d+)/30', line)
                    if match:
                        result["vocabulary_score"] = int(match.group(1))
                
                elif line.startswith("内容得分:"):
                    match = re.search(r'(\d+)/40', line)
                    if match:
                        result["content_score"] = int(match.group(1))
                
                # 提取评价
                elif line.startswith("总体评价:"):
                    result["overall_feedback"] = line.replace("总体评价:", "").strip()
                
                # 提取具体部分
                elif line.startswith("1. 优点:"):
                    current_section = "strengths"
                    content = line.replace("1. 优点:", "").strip()
                    if content:
                        result["strengths"].append(content)
                
                elif line.startswith("2. 需要改进的地方:"):
                    current_section = "weaknesses"
                    content = line.replace("2. 需要改进的地方:", "").strip()
                    if content:
                        result["weaknesses"].append(content)
                
                elif line.startswith("3. 具体建议:"):
                    current_section = "suggestions"
                    content = line.replace("3. 具体建议:", "").strip()
                    if content:
                        result["suggestions"].append(content)
                
                # 处理多行内容
                elif current_section and line and not line.startswith(("1.", "2.", "3.", "总体", "语法", "词汇", "内容")):
                    if current_section == "strengths":
                        result["strengths"].append(line)
                    elif current_section == "weaknesses":
                        result["weaknesses"].append(line)
                    elif current_section == "suggestions":
                        result["suggestions"].append(line)
                
                # 重置部分
                elif line.startswith(("1.", "2.", "3.", "总体", "语法", "词汇", "内容")):
                    current_section = None
            
        except Exception as e:
            print(f"解析响应出错: {e}")
        
        return result
    
    def check_grammar(self, text: str, return_dict: bool = False):
        """
        检查语法错误
        
        Args:
            text: 要检查的文本
            return_dict: 是否返回字典格式
            
        Returns:
            语法检查结果
        """
        if not self.grammar_tool:
            if return_dict:
                return {"errors": [], "error_count": 0, "message": "语法检查工具未启用"}
            return "语法检查工具未启用，请安装language-tool-python"
        
        try:
            matches = self.grammar_tool.check(text)
            
            if not matches:
                if return_dict:
                    return {"errors": [], "error_count": 0, "message": "未发现语法错误"}
                return "未发现语法错误。"
            
            errors = []
            for match in matches[:10]:  # 最多10个错误
                error_info = {
                    "error": match.message,
                    "suggestion": match.replacements[0] if match.replacements else "",
                    "context": match.context,
                    "offset": match.offset,
                    "length": match.errorLength
                }
                errors.append(error_info)
            
            if return_dict:
                return {
                    "errors": errors,
                    "error_count": len(matches),
                    "message": f"发现 {len(matches)} 个语法错误"
                }
            
            # 返回文本格式
            result = f"发现 {len(matches)} 个语法错误:\n\n"
            for i, error in enumerate(errors[:5], 1):  # 最多显示5个
                result += f"{i}. {error['error']}\n"
                if error['suggestion']:
                    result += f"   建议: {error['suggestion']}\n"
                result += f"   上下文: {error['context']}\n\n"
            
            return result
            
        except Exception as e:
            error_msg = f"语法检查失败: {e}"
            if return_dict:
                return {"errors": [], "error_count": 0, "message": error_msg}
            return error_msg
    
    def analyze_vocabulary(self, text: str, return_dict: bool = False):
        """
        分析词汇使用
        
        Args:
            text: 要分析的文本
            return_dict: 是否返回字典格式
            
        Returns:
            词汇分析结果
        """
        words = [word.lower() for word in text.split() if word.strip()]
        
        if not words:
            if return_dict:
                return {
                    "word_count": 0,
                    "unique_words": 0,
                    "lexical_diversity": 0,
                    "feedback": "文本为空"
                }
            return "文本为空"
        
        # 基本统计
        word_count = len(words)
        unique_words = set(words)
        unique_count = len(unique_words)
        lexical_diversity = unique_count / word_count
        
        # 词频分析
        from collections import Counter
        word_freq = Counter(words)
        common_words = word_freq.most_common(10)
        
        # 词汇级别分析
        basic_words = {
            'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'is', 'am', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did',
            'a', 'an', 'the', 'and', 'but', 'or', 'in',
            'on', 'at', 'to', 'for', 'of', 'with', 'by'
        }
        
        advanced_words = unique_words - basic_words
        advanced_ratio = len(advanced_words) / unique_count if unique_count > 0 else 0
        
        # 生成反馈
        feedback_parts = []
        
        if word_count < 30:
            feedback_parts.append("作文较短，建议扩展内容。")
        elif word_count > 200:
            feedback_parts.append("作文内容丰富，长度适中。")
        else:
            feedback_parts.append("作文长度合适。")
        
        if lexical_diversity < 0.4:
            feedback_parts.append("词汇重复较多，建议使用更多不同的词汇。")
        elif lexical_diversity > 0.7:
            feedback_parts.append("词汇多样性很好。")
        else:
            feedback_parts.append("词汇多样性一般，可以继续提高。")
        
        if advanced_ratio < 0.1:
            feedback_parts.append("可以尝试使用更多高级词汇。")
        elif advanced_ratio > 0.3:
            feedback_parts.append("高级词汇使用良好。")
        
        feedback = " ".join(feedback_parts)
        
        if return_dict:
            return {
                "word_count": word_count,
                "unique_words": unique_count,
                "lexical_diversity": round(lexical_diversity, 3),
                "advanced_ratio": round(advanced_ratio, 3),
                "common_words": common_words[:5],
                "feedback": feedback
            }
        
        # 返回文本格式
        result = f"词汇分析结果:\n"
        result += f"总单词数: {word_count}\n"
        result += f"独特单词数: {unique_count}\n"
        result += f"词汇多样性: {lexical_diversity:.1%}\n"
        result += f"高级词汇比例: {advanced_ratio:.1%}\n\n"
        result += f"最常见单词:\n"
        for word, freq in common_words[:5]:
            result += f"  {word}: {freq}次\n"
        result += f"\n建议: {feedback}"
        
        return result
    
    def _calculate_final_scores(self, llm_result: Dict, grammar_results: Dict, vocabulary_results: Dict) -> Dict[str, int]:
        """计算最终分数"""
        # 从大模型获取基础分数
        overall = llm_result.get("overall_score", 0)
        grammar = llm_result.get("grammar_score", 0)
        vocabulary = llm_result.get("vocabulary_score", 0)
        content = llm_result.get("content_score", 0)
        
        # 根据语法检查结果调整
        grammar_error_count = grammar_results.get("error_count", 0)
        if grammar_error_count > 10:
            grammar = max(0, grammar - 8)
        elif grammar_error_count > 5:
            grammar = max(0, grammar - 5)
        elif grammar_error_count > 2:
            grammar = max(0, grammar - 3)
        
        # 根据词汇多样性调整
        lexical_diversity = vocabulary_results.get("lexical_diversity", 0)
        if lexical_diversity < 0.3:
            vocabulary = max(0, vocabulary - 5)
        elif lexical_diversity > 0.6:
            vocabulary = min(30, vocabulary + 3)
        
        # 重新计算总分
        overall = int((grammar + vocabulary + content) * 0.98)  # 轻微调整
        
        return {
            "overall": overall,
            "grammar": grammar,
            "vocabulary": vocabulary,
            "content": content
        }
    
    def _get_empty_essay_result(self) -> Dict[str, Any]:
        """获取空作文的批改结果"""
        return {
            "overall_score": 0,
            "grammar_score": 0,
            "vocabulary_score": 0,
            "content_score": 0,
            "overall_feedback": "作文内容过短，请认真写作。",
            "grammar_errors": [],
            "suggestions": ["请写出至少20个单词的作文。"],
            "strengths": [],
            "weaknesses": ["内容过短"],
            "word_count": 0,
            "character_count": 0
        }
    
    def _get_default_grading_result(self) -> Dict[str, Any]:
        """获取默认批改结果"""
        return {
            "overall_score": 60,
            "grammar_score": 20,
            "vocabulary_score": 20,
            "content_score": 20,
            "overall_feedback": "批改系统暂时无法提供详细反馈。",
            "suggestions": [
                "请检查基本语法错误",
                "尝试使用更多样的词汇",
                "确保内容紧扣题目要求"
            ],
            "strengths": [],
            "weaknesses": []
        }

# 测试函数
def test_essay_grader():
    """测试作文批改器"""
    print("测试作文批改模块...")
    
    # 检查环境变量
    if not os.getenv("OPENAI_API_KEY"):
        print("请设置 OPENAI_API_KEY 环境变量")
        return
    
    # 创建批改器
    grader = EssayGrader(model_name="gpt-3.5-turbo")
    
    # 测试作文
    test_essay = """
    My family is very nice. I have a father, a mother and a sister. 
    My father is a teacher. He teaches math at a school. 
    My mother is a doctor. She helps sick people in hospital. 
    My sister is a student. She studies in primary school. 
    We like to watch TV together on weekends. I love my family.
    """
    
    test_prompt = {
        "title": "My Family",
        "prompt": "Write about your family members and what you like to do together."
    }
    
    print("测试作文批改...")
    result = grader.grade_essay(test_essay, test_prompt)
    
    print(f"总分: {result.get('overall_score', 0)}/100")
    print(f"语法: {result.get('grammar_score', 0)}/30")
    print(f"词汇: {result.get('vocabulary_score', 0)}/30")
    print(f"内容: {result.get('content_score', 0)}/40")
    print(f"字数: {result.get('word_count', 0)} 单词")
    
    feedback = result.get('overall_feedback', '')
    if feedback:
        print(f"\n总体评价: {feedback[:100]}...")
    
    suggestions = result.get('suggestions', [])
    if suggestions:
        print(f"\n建议 ({len(suggestions)} 条):")
        for i, suggestion in enumerate(suggestions[:3], 1):
            print(f"  {i}. {suggestion}")
    
    # 测试语法检查
    print("\n测试语法检查...")
    grammar_text = "I goes to school everyday. He don't like apple. She have two book."
    grammar_result = grader.check_grammar(grammar_text)
    print(f"语法检查结果: {grammar_result[:100]}...")
    
    # 测试词汇分析
    print("\n测试词汇分析...")
    vocab_text = "My family is very important to me. We have three people in our family. We like to spend time together."
    vocab_result = grader.analyze_vocabulary(vocab_text)
    print(f"词汇分析结果: {vocab_result[:100]}...")
    
    print("\n✅ 作文批改测试完成")

if __name__ == "__main__":
    test_essay_grader()

