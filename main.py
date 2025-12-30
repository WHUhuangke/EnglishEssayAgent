"""
中小学英语作文出题和批改智能体 - 主程序入口
使用LangChain框架实现模块化系统
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 加载环境变量
load_dotenv()

# 导入自定义模块
from knowledge_base import EssayKnowledgeBase
from essay_grader import EssayGrader
from agent_workflow import EssayAgentWorkflow
from config import MODEL_CONFIG, GRADE_LEVELS, ESSAY_GENRES

class EnglishEssayAgent:
    """英语作文智能体主类"""
    
    def __init__(self, model_name: str = None):
        """
        初始化智能体
        
        Args:
            model_name: 使用的模型名称，默认为配置文件中的设置
        """
        if model_name is None:
            model_name = MODEL_CONFIG.get("default_model", "gpt-3.5-turbo")
        
        # 检查API密钥
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  警告: 未设置 OPENAI_API_KEY 环境变量")
            print("请在 .env 文件中设置: OPENAI_API_KEY='your-key-here'")
            print("或在运行前执行: export OPENAI_API_KEY='your-key-here'")
            self.initialized = False
            return
        
        try:
            # 初始化知识库
            self.knowledge_base = EssayKnowledgeBase()
            print('knowledge_base init sucess')
            # 初始化批改器
            self.essay_grader = EssayGrader(model_name=model_name)
            print('grader init sucess')
            # 初始化工作流
            self.workflow = EssayAgentWorkflow(
                knowledge_base=self.knowledge_base,
                essay_grader=self.essay_grader
            )
            
            self.initialized = True
            print("✓ 英语作文智能体初始化完成")
            
        except Exception as e:
            print(f"❌ 智能体初始化失败: {e}")
            self.initialized = False
    
    def run_interactive(self):
        """运行交互式对话界面"""
        if not self.initialized:
            print("智能体未初始化，无法运行")
            return
        
        print("=" * 60)
        print("    中小学英语作文出题和批改智能体")
        print("=" * 60)
        print("功能说明:")
        print("1. 系统会根据您的年级和学习情况推荐作文题目")
        print("2. 您可以根据文体或主题进行筛选")
        print("3. 完成作文后，系统会自动批改和评分")
        print("4. 输入 'quit' 或 'exit' 退出程序")
        print("=" * 60)
        
        while True:
            try:
                # 第一步：获取用户信息
                print("\n" + "-" * 40)
                print("请提供以下信息 (或输入 'quit' 退出):")
                
                # 年级选择
                print("📅 年级选择:")
                print("1. 小学（默认）")
                print("2. 初中")
                print("3. 高中")
                grade_choice = input("请选择 (1/2/3): ").strip()
                if grade_choice.lower() in ['quit', 'exit', 'q']:
                    print("感谢使用，再见！")
                    break

                grade_map = {'1': '小学', '2': '初中', '3': '高中'}
                grade = grade_map.get(grade_choice, '小学')

                # 英语水平选择
                print("\n📊 英语水平选择:")
                print("1. 初级（默认）")
                print("2. 中级")
                print("3. 高级")
                level_choice = input("请选择 (1/2/3): ").strip().lower()
                if level_choice.lower() in ['quit', 'exit', 'q']:
                    print("感谢使用，再见！")
                    break

                level_map = {'1': 'beginner', '2': 'intermediate', '3': 'advanced'}
                level = level_map.get(level_choice, 'intermediate')
                
                # 显示可选的文体
                print("\n可选文体:")
                for eng, chi in ESSAY_GENRES.items():
                    print(f"  {eng}: {chi}")
                
                genre = input("\n✍️  想要练习的文体 (可选，直接回车跳过): ").strip()
                if genre.lower() in ['quit', 'exit', 'q']:
                    print("感谢使用，再见！")
                    break
                
                topic = input("🔍 想要练习的主题 (可选，直接回车跳过): ").strip()
                if topic.lower() in ['quit', 'exit', 'q']:
                    print("感谢使用，再见！")
                    break
                
                if not genre:
                    genre = None
                if not topic:
                    topic = None
                print('grade:',grade,' level:',level,' genre:',genre,' topic:',topic)
                # 第二步：搜索题目
                print("\n🔍 正在搜索合适的作文题目...")
                selected_prompt = self.workflow.search_essay_prompt(
                    grade=grade,
                    level=level,
                    genre=genre,
                    topic=topic
                )
                
                if not selected_prompt:
                    print("⚠️  没有找到完全匹配的作文题目，将使用相关题目")
                    # 尝试放宽条件
                    selected_prompt = self.workflow.search_essay_prompt(
                        grade=grade,
                        level=level,
                        genre=None,
                        topic=None
                    )
                
                if not selected_prompt:
                    print("❌ 无法找到合适的作文题目，请重新尝试")
                    continue
                
                # 展示题目
                print(f"\n📝 作文题目: {selected_prompt.get('title', 'My Essay')}")
                print(f"📄 写作要求: {selected_prompt.get('prompt', '')}")
                
                requirements = selected_prompt.get('requirements', [])
                if requirements:
                    print("📋 具体要求:")
                    for i, req in enumerate(requirements, 1):
                        print(f"  {i}. {req}")
                
                # 显示字数建议
                grade_key = self.workflow._normalize_grade(grade)
                if grade_key in GRADE_LEVELS:
                    config = GRADE_LEVELS[grade_key]
                    print(f"\n💡 字数建议: {config.get('min_words', 50)}-{config.get('max_words', 100)} 个单词")
                
                # 第三步：获取用户作文
                print("\n" + "=" * 40)
                print("请开始写作 (完成后，在新的一行输入 'END' 并回车):")
                essay_lines = []
                line_count = 0
                
                while True:
                    try:
                        line = input()
                        if line.strip().upper() == 'END':
                            break
                        essay_lines.append(line)
                        line_count += 1
                        
                        # 每5行显示一次进度
                        if line_count % 5 == 0:
                            word_count = len(' '.join(essay_lines).split())
                            print(f"  当前字数: {word_count} 单词")
                            
                    except EOFError:
                        print("\n检测到输入结束")
                        break
                    except KeyboardInterrupt:
                        print("\n\n输入被中断")
                        return
                
                essay = '\n'.join(essay_lines)
                
                if not essay.strip():
                    print("⚠️  作文内容为空，请重新开始")
                    continue
                
                # 第四步：批改作文
                print("\n⏳ 正在批改作文，请稍候...")
                grading_result = self.workflow.grade_essay(
                    essay=essay,
                    prompt_info=selected_prompt
                )
                
                # 显示批改结果
                self._display_grading_result(grading_result)
                
                # 询问是否继续
                print("\n" + "=" * 40)
                continue_choice = input("是否继续练习? (yes/no): ").strip().lower()
                if continue_choice not in ['yes', 'y', '是', '继续']:
                    print("感谢使用，再见！")
                    break
                    
            except KeyboardInterrupt:
                print("\n\n程序被用户中断")
                break
            except Exception as e:
                print(f"❌ 发生错误: {e}")
                print("请重新开始...")
    
    def _display_grading_result(self, result: dict):
        """显示批改结果"""
        print("\n" + "=" * 60)
        print("                 作文批改结果")
        print("=" * 60)
        
        # 显示分数
        print(f"📊 总分: {result.get('overall_score', 0)}/100")
        print(f"  🔤 语法得分: {result.get('grammar_score', 0)}/30")
        print(f"  📖 词汇得分: {result.get('vocabulary_score', 0)}/30")
        print(f"  📋 内容得分: {result.get('content_score', 0)}/40")
        
        # 显示字数统计
        print(f"\n📈 字数统计:")
        print(f"  单词数: {result.get('word_count', 0)}")
        print(f"  字符数: {result.get('character_count', 0)}")
        
        # 显示总体评价
        feedback = result.get('overall_feedback', '')
        if feedback:
            print(f"\n📝 总体评价:")
            print(f"  {feedback}")
        
        # 显示语法错误
        grammar_errors = result.get('grammar_errors', [])
        if grammar_errors:
            print(f"\n❌ 语法错误 ({len(grammar_errors)} 处):")
            for i, error in enumerate(grammar_errors[:5], 1):  # 最多显示5个
                if isinstance(error, dict):
                    error_msg = error.get('error', str(error))
                else:
                    error_msg = str(error)
                print(f"  {i}. {error_msg}")
        
        # 显示改进建议
        suggestions = result.get('suggestions', [])
        if suggestions:
            print(f"\n💡 改进建议 ({len(suggestions)} 条):")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"  {i}. {suggestion}")
    
    def test_grammar_checker(self):
        """测试语法检查工具"""
        if not self.initialized:
            print("智能体未初始化")
            return
        
        print("\n" + "=" * 40)
        print("语法检查工具测试")
        print("=" * 40)
        
        test_text = "I goes to school everyday. He don't like apple. She have two book."
        print(f"测试文本: {test_text}")
        
        result = self.essay_grader.check_grammar(test_text)
        print(f"\n检查结果: {result}")
    
    def test_vocabulary_analyzer(self):
        """测试词汇分析工具"""
        if not self.initialized:
            print("智能体未初始化")
            return
        
        print("\n" + "=" * 40)
        print("词汇分析工具测试")
        print("=" * 40)
        
        test_text = "My family is very important to me. We have three people in our family. We like to spend time together."
        print(f"测试文本: {test_text}")
        
        result = self.essay_grader.analyze_vocabulary(test_text)
        print(f"\n分析结果: {result}")

def main():
    """主函数"""
    # 创建智能体实例
    agent = EnglishEssayAgent()
    
    if not agent.initialized:
        return
    
    # 运行交互式界面
    agent.run_interactive()

def test():
    """测试函数"""
    print("运行测试...")
    
    # 创建智能体实例
    agent = EnglishEssayAgent()
    
    if not agent.initialized:
        print("测试失败: 智能体未初始化")
        return
    
    # 测试语法检查
    agent.test_grammar_checker()
    
    # 测试词汇分析
    agent.test_vocabulary_analyzer()
    
    print("\n✅ 测试完成")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='中小学英语作文出题和批改智能体')
    parser.add_argument('--test', action='store_true', help='运行测试')
    
    args = parser.parse_args()
    
    if args.test:
        test()
    else:
        main()

