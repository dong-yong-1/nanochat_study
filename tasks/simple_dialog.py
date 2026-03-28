"""
Simple Dialog and QA Evaluation Task for Small Models.
Designed for models with limited capacity (e.g., 12 layers, 512 dims).
Focuses on basic conversation and instruction following abilities.
"""

from tasks.common import Task

class SimpleDialog(Task):
    """
    Simple dialog and QA evaluation task for small models.
    Contains basic conversation patterns and simple questions.
    """

    def __init__(self, split="test", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SimpleDialog split must be train|test"
        self.split = split
        
        # Define simple test cases
        self.test_cases = self._create_test_cases()
        self.length = len(self.test_cases)

    def _create_test_cases(self):
        """Create simple test cases suitable for small models."""
        
        # Basic greetings and responses
        greetings = [
            {
                "messages": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "你好！很高兴见到你。"}
                ],
                "evaluation_keywords": ["你好", "高兴", "见到"]
            },
            {
                "messages": [
                    {"role": "user", "content": "早上好"},
                    {"role": "assistant", "content": "早上好！祝你今天有个美好的一天。"}
                ],
                "evaluation_keywords": ["早上好", "美好", "一天"]
            },
            {
                "messages": [
                    {"role": "user", "content": "再见"},
                    {"role": "assistant", "content": "再见！期待下次和你聊天。"}
                ],
                "evaluation_keywords": ["再见", "期待", "聊天"]
            }
        ]
        
        # Simple identity questions
        identity_questions = [
            {
                "messages": [
                    {"role": "user", "content": "你叫什么名字？"},
                    {"role": "assistant", "content": "我是一个AI助手，没有特定的名字。"}
                ],
                "evaluation_keywords": ["AI", "助手", "名字"]
            },
            {
                "messages": [
                    {"role": "user", "content": "你能做什么？"},
                    {"role": "assistant", "content": "我可以回答问题、进行对话、提供信息等。"}
                ],
                "evaluation_keywords": ["回答", "对话", "信息"]
            }
        ]
        
        # Simple factual questions
        factual_questions = [
            {
                "messages": [
                    {"role": "user", "content": "1+1等于多少？"},
                    {"role": "assistant", "content": "1+1等于2。"}
                ],
                "evaluation_keywords": ["2"]
            },
            {
                "messages": [
                    {"role": "user", "content": "一年有多少个月？"},
                    {"role": "assistant", "content": "一年有12个月。"}
                ],
                "evaluation_keywords": ["12"]
            },
            {
                "messages": [
                    {"role": "user", "content": "一周有几天？"},
                    {"role": "assistant", "content": "一周有7天。"}
                ],
                "evaluation_keywords": ["7"]
            },
            {
                "messages": [
                    {"role": "user", "content": "太阳从哪个方向升起？"},
                    {"role": "assistant", "content": "太阳从东方升起。"}
                ],
                "evaluation_keywords": ["东方"]
            }
        ]
        
        # Simple instruction following
        instructions = [
            {
                "messages": [
                    {"role": "user", "content": "请说'好的'"},
                    {"role": "assistant", "content": "好的"}
                ],
                "evaluation_keywords": ["好的"]
            },
            {
                "messages": [
                    {"role": "user", "content": "请重复：今天天气很好"},
                    {"role": "assistant", "content": "今天天气很好"}
                ],
                "evaluation_keywords": ["今天", "天气", "很好"]
            },
            {
                "messages": [
                    {"role": "user", "content": "用'因为...所以...'造句"},
                    {"role": "assistant", "content": "因为下雨了，所以地面湿了。"}
                ],
                "evaluation_keywords": ["因为", "所以"]
            }
        ]
        
        # Simple conversation
        conversations = [
            {
                "messages": [
                    {"role": "user", "content": "你喜欢什么颜色？"},
                    {"role": "assistant", "content": "作为AI，我没有个人喜好，但我可以帮你了解各种颜色。"}
                ],
                "evaluation_keywords": ["AI", "喜好", "颜色"]
            },
            {
                "messages": [
                    {"role": "user", "content": "今天怎么样？"},
                    {"role": "assistant", "content": "我很好，谢谢关心！有什么我可以帮你的吗？"}
                ],
                "evaluation_keywords": ["好", "帮"]
            }
        ]
        
        # Combine all test cases
        all_cases = greetings + identity_questions + factual_questions + instructions + conversations
        
        return all_cases

    def num_examples(self):
        return self.length

    def get_example(self, index):
        return self.test_cases[index]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.get_example(index)

    @property
    def eval_type(self):
        return 'generative'

    def evaluate(self, conversation, completion):
        """
        Evaluate if the completion is a valid response.
        
        Args:
            conversation: The conversation context
            completion: The model's generated response
            
        Returns:
            bool: True if the response is valid, False otherwise
        """
        # Get the expected keywords for this conversation
        expected_keywords = conversation.get("evaluation_keywords", [])
        
        # Check if any expected keyword appears in the completion
        if not expected_keywords:
            return True  # No specific keywords required
            
        completion_lower = completion.lower()
        keyword_matches = sum(1 for keyword in expected_keywords if keyword in completion_lower)
        
        # Consider it valid if at least 50% of keywords are present
        min_required = max(1, len(expected_keywords) // 2)
        return keyword_matches >= min_required