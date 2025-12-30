"""
配置模块
包含系统配置、常量定义和工具函数
"""
import os
from typing import Dict, Any, List
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
VECTOR_DB_DIR = DATA_DIR / "chroma_db"

# 创建必要的目录
DATA_DIR.mkdir(exist_ok=True)
VECTOR_DB_DIR.mkdir(exist_ok=True)

# 模型配置
MODEL_CONFIG = {
    "default_model": "gpt-5-mini-2025-08-07",
    "embedding_model": "text-embedding-3-small",
    "temperature": 0.3,
    "max_tokens": 2000
}

# 向量知识库配置
VECTOR_STORE_CONFIG = {
    "persist_directory": str(VECTOR_DB_DIR),
    "collection_name": "essay_prompts",
    "embedding_dimension": 1536
}

# 评分配置
GRADING_CONFIG = {
    "grammar_weight": 0.3,
    "vocabulary_weight": 0.3,
    "content_weight": 0.4,
    "min_word_count": 20,
    "max_word_count": 500
}

# 年级配置
GRADE_LEVELS = {
    "primary": {          # 小学
        "min_words": 30,
        "max_words": 100,
        "level": "beginner",
        "description": "小学"
    },
    "middle": {           # 初中
        "min_words": 80,
        "max_words": 180,
        "level": "intermediate",
        "description": "初中"
    },
    "high": {             # 高中
        "min_words": 150,
        "max_words": 300,
        "level": "advanced",
        "description": "高中"
    }
}

# 年级名称映射
GRADE_NAME_MAPPING = {
    # 中文
    "小学": "primary_school",
    "初中": "middle_school",
    "高中": "high_school",
    # 英文
    "primary": "primary_school",
    "middle": "middle_school",
    "high": "high_school",
}
# 水平映射
LEVEL_MAPPING = {
    "初级": "beginner",
    "中级": "intermediate",
    "高级": "advanced",
    "beginner": "beginner",
    "intermediate": "intermediate",
    "advanced": "advanced",
    "基础": "beginner",
    "中等": "intermediate",
    "进阶": "advanced"
}

# 文体类型
ESSAY_GENRES = {
    "narrative": "记叙文",
    "descriptive": "描写文",
    "argumentative": "议论文",
    "letter": "书信",
    "opinion": "观点文",
    "expository": "说明文",
    "diary": "日记",
    "story": "故事",
    "report": "报告",
    "email": "邮件"
}

# 主题分类
ESSAY_TOPICS = {
    "family": "家庭",
    "school": "学校",
    "friend": "朋友",
    "hobby": "爱好",
    "travel": "旅行",
    "environment": "环境",
    "technology": "科技",
    "sport": "运动",
    "food": "食物",
    "festival": "节日",
    "animal": "动物",
    "book": "书籍",
    "movie": "电影",
    "music": "音乐",
    "dream": "梦想"
}

# 示例作文题目
SAMPLE_ESSAY_PROMPTS = [
    {
        "id": 1,
        "title": "My Family",
        "prompt": "Write a short essay about your family. Describe your family members and what you like to do together. (60-80 words)",
        "grade": "primary_school_5",
        "level": "beginner",
        "genre": "narrative",
        "topic": "family",
        "requirements": [
            "Use simple present tense",
            "Include at least 3 family members",
            "Use adjectives to describe people"
        ],
        "keywords": ["family", "parents", "siblings", "love", "home"]
    },
    {
        "id": 2,
        "title": "My Hometown",
        "prompt": "Describe your hometown. What does it look like? What are the special places there? (80-100 words)",
        "grade": "middle_school_2",
        "level": "intermediate",
        "genre": "descriptive",
        "topic": "hometown",
        "requirements": [
            "Use descriptive adjectives",
            "Describe at least 2 places",
            "Use there is/are structure"
        ],
        "keywords": ["hometown", "city", "countryside", "places", "description"]
    },
    {
        "id": 3,
        "title": "The Advantages of Technology",
        "prompt": "Write an essay about the advantages of technology in our daily life. Give at least 3 examples. (100-120 words)",
        "grade": "high_school_1",
        "level": "advanced",
        "genre": "argumentative",
        "topic": "technology",
        "requirements": [
            "Use topic sentences",
            "Provide clear examples",
            "Use connectors: firstly, secondly, finally"
        ],
        "keywords": ["technology", "advantages", "internet", "communication", "education"]
    },
    {
        "id": 4,
        "title": "A Letter to My Friend",
        "prompt": "Write a letter to your friend. Tell them about your weekend and ask about theirs. (50-70 words)",
        "grade": "primary_school_4",
        "level": "beginner",
        "genre": "letter",
        "topic": "friend",
        "requirements": [
            "Use letter format (Dear..., Love...)",
            "Write about your weekend",
            "Ask 2 questions"
        ],
        "keywords": ["letter", "friend", "weekend", "share", "questions"]
    },
    {
        "id": 5,
        "title": "How to Protect the Environment",
        "prompt": "Give your opinion on how we can protect the environment. Suggest at least 3 ways. (120-150 words)",
        "grade": "high_school_2",
        "level": "advanced",
        "genre": "opinion",
        "topic": "environment",
        "requirements": [
            "State your opinion clearly",
            "Use persuasive language",
            "Provide supporting reasons"
        ],
        "keywords": ["environment", "protection", "pollution", "solutions", "earth"]
    }
]

def get_grade_config(grade: str) -> Dict[str, Any]:
    """
    获取年级配置
    
    Args:
        grade: 年级标识
        
    Returns:
        年级配置字典
    """
    # 首先尝试直接匹配
    if grade in GRADE_LEVELS:
        return GRADE_LEVELS[grade]
    
    # 尝试通过映射匹配
    normalized_grade = GRADE_NAME_MAPPING.get(grade)
    if normalized_grade and normalized_grade in GRADE_LEVELS:
        return GRADE_LEVELS[normalized_grade]
    
    # 返回默认配置
    return {
        "min_words": 80,
        "max_words": 150,
        "level": "intermediate",
        "description": "默认年级"
    }

def normalize_grade(grade: str) -> str:
    """
    标准化年级标识
    
    Args:
        grade: 原始年级字符串
        
    Returns:
        标准化后的年级标识
    """
    if grade in GRADE_LEVELS:
        return grade
    
    normalized = GRADE_NAME_MAPPING.get(grade)
    if normalized and normalized in GRADE_LEVELS:
        return normalized
    
    # 如果无法识别，返回默认
    return "middle_school_2"

def normalize_level(level: str) -> str:
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

def validate_essay_length(essay: str, grade: str) -> Dict[str, Any]:
    """
    验证作文长度是否符合年级要求
    
    Args:
        essay: 作文文本
        grade: 年级标识
        
    Returns:
        验证结果字典
    """
    word_count = len(essay.split())
    config = get_grade_config(grade)
    
    min_words = config.get("min_words", 50)
    max_words = config.get("max_words", 150)
    
    is_valid = min_words <= word_count <= max_words
    
    feedback = ""
    if word_count < min_words:
        feedback = f"作文过短，建议至少写{min_words}个单词。"
    elif word_count > max_words:
        feedback = f"作文过长，建议不超过{max_words}个单词。"
    else:
        feedback = f"作文长度合适({word_count}个单词)。"
    
    return {
        "is_valid": is_valid,
        "word_count": word_count,
        "min_words": min_words,
        "max_words": max_words,
        "feedback": feedback
    }

def load_environment_variables() -> bool:
    """
    加载环境变量并检查必要变量
    
    Returns:
        是否所有必要环境变量都已设置
    """
    from dotenv import load_dotenv
    load_dotenv()
    
    # 检查必要的环境变量
    required_vars = ['OPENAI_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("⚠️  警告: 以下环境变量未设置:")
        for var in missing_vars:
            print(f"  - {var}")
        print("请在 .env 文件中设置这些变量")
        return False
    
    return True

if __name__ == "__main__":
    # 测试配置函数
    print("测试配置模块...")
    
    # 测试年级配置获取
    test_grade = "初中二年级"
    config = get_grade_config(test_grade)
    print(f"{test_grade} 配置: {config}")
    
    # 测试标准化
    normalized = normalize_grade(test_grade)
    print(f"标准化后的年级: {normalized}")
    
    # 测试作文长度验证
    test_essay = "This is a test essay with several words to test the length validation function."
    validation = validate_essay_length(test_essay, test_grade)
    print(f"作文长度验证: {validation}")
    
    # 测试环境变量加载
    env_loaded = load_environment_variables()
    print(f"环境变量加载: {'成功' if env_loaded else '失败'}")

