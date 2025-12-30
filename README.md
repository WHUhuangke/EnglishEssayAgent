# 中小学英语作文出题和批改智能体

## 项目概述

这是一个基于LangChain框架构建的中小学英语作文出题和批改智能体系统。系统通过RAG技术从向量知识库中检索合适的作文题目，并使用大模型结合自定义工具对学生的作文进行多维度批改和评分。

## 主要功能

- **智能出题**: 根据年级、英语水平、文体和主题推荐合适的作文题目
- **自动批改**: 对提交的作文进行语法检查、词汇分析和内容评估
- **详细反馈**: 提供分数、错误分析和改进建议
- **个性化推荐**: 根据学生水平匹配合适难度的题目
- **进度跟踪**: 记录和分析学习进度

## 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                    用户交互层                                │
│                    (CLI界面)                                │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                    智能体工作流层                            │
│               (AgentWorkflow - 协调所有组件)                 │
└───────────────┬─────────────┬───────────────┬───────────────┘
                │             │               │
    ┌───────────▼──────┐ ┌───▼─────────┐ ┌───▼─────────┐
    │  知识库管理      │ │  作文批改    │ │  自定义工具  │
    │  (RAG检索)       │ │  (多维度评估)│ │  (语法检查)  │
    └──────────────────┘ └──────────────┘ └──────────────┘
                │               │               │
    ┌───────────▼──────┐ ┌───▼─────────┐ ┌───▼─────────┐
    │  向量数据库      │ │  LLM大模型   │ │  词汇分析    │
    │  (ChromaDB)      │ │  (GPT-3.5)  │ │  (本地算法)  │
    └──────────────────┘ └──────────────┘ └──────────────┘
```

## 安装指南

### 系统要求

- Python 3.8+
- OpenAI API密钥

### 环境设置

1. **克隆项目**
```bash
git clone <repository-url>
cd english_essay_agent
```

2. **创建虚拟环境** (推荐)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **配置环境变量**
```bash
cp .env.example .env
# 编辑.env文件，添加您的OpenAI API密钥
```

`.env`文件内容示例：
```env
# OpenAI API配置
OPENAI_API_KEY=sk-your-openai-api-key-here

# 可选：其他配置
# CHROMA_DB_PATH=./data/chroma_db
# MODEL_NAME=gpt-3.5-turbo
```

## 快速开始

### 1. 基本使用

```bash
# 启动智能体
python main.py
```

系统启动后，按照提示输入信息：
1. 输入年级 (如: 初中二年级)
2. 输入英语水平 (beginner/intermediate/advanced)
3. 输入想要练习的文体 (可选，如: narrative/descriptive)
4. 输入想要练习的主题 (可选，如: family/technology)
5. 根据系统提供的题目进行写作
6. 输入'END'完成作文
7. 查看批改结果

### 2. 测试模式

```bash
# 运行测试模式
python main.py --test
```

测试模式会验证所有组件是否正常工作。

### 3. 命令行参数

```bash
# 指定模型
python main.py --model gpt-4

# 调试模式
python main.py --verbose

# 查看帮助
python main.py --help
```

## 模块说明

### 1. 主程序 (main.py)

智能体的入口点，负责用户交互和流程控制。

主要功能：
- 收集用户输入信息
- 调用工作流模块
- 显示题目和批改结果
- 管理用户会话

### 2. 配置文件 (config.py)

包含系统配置和常量定义：
- 模型配置 (MODEL_CONFIG)
- 年级和水平映射 (GRADE_LEVELS)
- 文体类型定义 (ESSAY_GENRES)
- 主题分类 (ESSAY_TOPICS)
- 评分配置 (GRADING_CONFIG)

### 3. 知识库模块 (knowledge_base.py)

管理作文题目的向量存储和检索：

```python
# 初始化知识库
kb = EssayKnowledgeBase()

# 添加题目
prompt_data = {
    "title": "My Family",
    "prompt": "Write about your family...",
    "grade": "primary_school_5",
    "level": "beginner",
    "genre": "narrative",
    "topic": "family"
}
kb.add_essay_prompt(prompt_data)

# 搜索题目
prompts = kb.search_prompts(
    grade="middle_school_2",
    level="intermediate",
    genre="narrative",
    topic="family"
)
```

### 4. 作文批改模块 (essay_grader.py)

负责作文的批改和评分：

```python
# 初始化批改器
grader = EssayGrader(model_name="gpt-3.5-turbo")

# 批改作文
result = grader.grade_essay(essay_text, prompt_info)

# 获取批改结果
print(f"总分: {result['overall_score']}/100")
print(f"语法得分: {result['grammar_score']}/30")
print(f"词汇得分: {result['vocabulary_score']}/30")
print(f"内容得分: {result['content_score']}/40")
```

### 5. 智能体工作流 (agent_workflow.py)

协调整个系统的运行流程：

```python
# 初始化工作流
workflow = EssayAgentWorkflow(knowledge_base=kb, essay_grader=grader)

# 搜索题目
prompt = workflow.search_essay_prompt(
    grade="初中二年级",
    level="intermediate",
    genre="narrative"
)

# 批改作文
grading_result = workflow.grade_essay(essay, prompt)
```

### 6. 自定义工具 (tools.py)

扩展功能的自定义工具：
- GrammarCheckerTool: 语法检查工具
- VocabularyAnalyzerTool: 词汇分析工具

## API接口说明

### 作文题目检索API

```python
# 根据条件检索作文题目
def search_essay_prompt(grade: str, level: str, genre: str = None, topic: str = None)
"""
参数:
  grade: 年级 (如: "primary_school_5", "middle_school_2")
  level: 水平 (如: "beginner", "intermediate", "advanced")
  genre: 文体类型 (可选)
  topic: 主题 (可选)

返回:
  作文题目信息字典
"""
```

### 作文批改API

```python
# 批改作文并返回详细结果
def grade_essay(essay: str, prompt_info: dict) -> dict
"""
参数:
  essay: 学生作文文本
  prompt_info: 作文题目信息

返回:
  批改结果字典，包含:
    - overall_score: 总分
    - grammar_score: 语法得分
    - vocabulary_score: 词汇得分
    - content_score: 内容得分
    - overall_feedback: 总体评价
    - grammar_errors: 语法错误列表
    - suggestions: 改进建议
"""
```

## 自定义配置

### 1. 添加新的作文题目

编辑`config.py`文件中的`SAMPLE_ESSAY_PROMPTS`列表：

```python
SAMPLE_ESSAY_PROMPTS = [
    {
        "id": 6,
        "title": "My Favorite Season",
        "prompt": "Write an essay about your favorite season and explain why you like it. (70-90 words)",
        "grade": "primary_school_4",
        "level": "beginner",
        "genre": "descriptive",
        "topic": "season",
        "requirements": [
            "Describe the weather",
            "Mention seasonal activities",
            "Use adjectives"
        ],
        "keywords": ["season", "weather", "activities", "favorite"]
    },
    # 添加更多题目...
]
```

### 2. 调整评分标准

在`config.py`中修改`GRADING_CONFIG`：

```python
GRADING_CONFIG = {
    "grammar_weight": 0.35,      # 增加语法权重
    "vocabulary_weight": 0.25,   # 调整词汇权重
    "content_weight": 0.40,      # 内容权重
    "min_word_count": 25,        # 最小单词数
    "max_word_count": 500        # 最大单词数
}
```

### 3. 配置年级要求

在`config.py`中修改`GRADE_LEVELS`：

```python
GRADE_LEVELS = {
    "primary_school_1": {
        "min_words": 30,
        "max_words": 50,
        "level": "beginner",
        "description": "小学一年级"
    },
    # 调整其他年级...
}
```

## 数据管理

### 1. 初始化知识库

系统首次运行时会自动创建包含示例题目的向量知识库。数据存储在`data/chroma_db`目录中。

### 2. 备份和恢复

```python
# 备份知识库
knowledge_base.save_to_json("backup.json")

# 从备份恢复
knowledge_base.load_from_json("backup.json")
```

### 3. 查看知识库内容

```python
# 获取所有题目
all_prompts = knowledge_base.get_all_prompts()
print(f"知识库中有 {len(all_prompts)} 个作文题目")

# 获取题目数量
count = knowledge_base.get_prompt_count()
print(f"题目数量: {count}")
```

## 故障排除

### 常见问题

1. **OpenAI API密钥错误**
   ```
   错误: 请设置 OPENAI_API_KEY 环境变量
   解决: 在.env文件中正确设置API密钥
   ```

2. **知识库初始化失败**
   ```
   错误: 无法连接到向量数据库
   解决: 确保有写入data/chroma_db目录的权限
   ```

3. **语法检查工具不可用**
   ```
   警告: 未安装language-tool-python
   解决: pip install language-tool-python
   ```

4. **内存不足**
   ```
   错误: ChromaDB内存不足
   解决: 减少知识库中的题目数量或增加系统内存
   ```

### 调试模式

启用调试模式查看更多信息：

```bash
python main.py --verbose
```

## 扩展开发

### 1. 添加新的文体类型

编辑`config.py`中的`ESSAY_GENRES`字典：

```python
ESSAY_GENRES = {
    # 现有文体...
    "new_genre": "新文体名称",
}
```

### 2. 添加新的工具

在`tools.py`中添加新的工具类：

```python
class NewAnalysisTool(BaseTool):
    name = "new_analysis"
    description = "新的分析工具描述"
    
    def _run(self, input_text: str) -> str:
        # 实现工具逻辑
        return "分析结果"
```

### 3. 自定义批改逻辑

在`essay_grader.py`中扩展批改功能：

```python
def custom_grading_method(self, essay: str) -> dict:
    """自定义批改方法"""
    # 实现自定义批改逻辑
    return {
        "custom_score": 90,
        "custom_feedback": "自定义反馈"
    }
```

## 性能优化

### 1. 缓存优化

系统使用向量数据库进行语义搜索，建议：
- 定期清理不需要的向量数据
- 使用合适的嵌入模型尺寸
- 调整相似度搜索的k值

### 2. 批处理

对于批量处理，可以：
- 使用异步处理提高效率
- 实现批量作文批改功能
- 缓存频繁使用的题目

### 3. 资源管理

- 最小化内存使用：定期清理缓存
- 优化API调用：合并多次调用
- 使用更轻量的模型：在配置文件中切换模型

## 贡献指南

1. Fork项目仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开Pull Request

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 技术支持

如有问题或建议，请：
1. 查看docs/FAQ.md
2. 提交https://github.com/your-repo/issues
3. 联系项目维护者

---

**版本**: 1.0.0  
**最后更新**: 2024年1月  
**维护者**: 项目团队  
**联系方式**: your-email@example.com
