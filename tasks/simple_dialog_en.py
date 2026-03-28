"""
Simple Dialog and QA Evaluation Task for Small Models (English version).
Designed for models with limited capacity (e.g., 12 layers, 512 dims).
Focuses on basic conversation and instruction following abilities.
"""

from tasks.common import Task

class SimpleDialogEN(Task):
    """
    Simple dialog and QA evaluation task for small models (English version).
    Contains basic conversation patterns and simple questions in English.
    """

    def __init__(self, split="test", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SimpleDialogEN split must be train|test"
        self.split = split
        
        # Define simple test cases in English
        self.test_cases = self._create_test_cases()
        self.length = len(self.test_cases)

    def _create_test_cases(self):
        """Create simple test cases in English suitable for small models."""
        
        # Basic greetings and responses
        greetings = [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hello! Nice to meet you."}
                ],
                "evaluation_keywords": ["hello", "nice", "meet"]
            },
            {
                "messages": [
                    {"role": "user", "content": "Good morning"},
                    {"role": "assistant", "content": "Good morning! Have a wonderful day."}
                ],
                "evaluation_keywords": ["good", "morning", "wonderful"]
            },
            {
                "messages": [
                    {"role": "user", "content": "Goodbye"},
                    {"role": "assistant", "content": "Goodbye! Looking forward to chatting with you again."}
                ],
                "evaluation_keywords": ["goodbye", "looking", "forward"]
            }
        ]
        
        # Simple identity questions
        identity_questions = [
            {
                "messages": [
                    {"role": "user", "content": "What's your name?"},
                    {"role": "assistant", "content": "I'm an AI assistant, I don't have a specific name."}
                ],
                "evaluation_keywords": ["AI", "assistant", "name"]
            },
            {
                "messages": [
                    {"role": "user", "content": "What can you do?"},
                    {"role": "assistant", "content": "I can answer questions, have conversations, provide information, etc."}
                ],
                "evaluation_keywords": ["answer", "conversations", "information"]
            }
        ]
        
        # Simple factual questions
        factual_questions = [
            {
                "messages": [
                    {"role": "user", "content": "What's 1+1?"},
                    {"role": "assistant", "content": "1+1 equals 2."}
                ],
                "evaluation_keywords": ["2"]
            },
            {
                "messages": [
                    {"role": "user", "content": "How many months are there in a year?"},
                    {"role": "assistant", "content": "There are 12 months in a year."}
                ],
                "evaluation_keywords": ["12"]
            },
            {
                "messages": [
                    {"role": "user", "content": "How many days are there in a week?"},
                    {"role": "assistant", "content": "There are 7 days in a week."}
                ],
                "evaluation_keywords": ["7"]
            },
            {
                "messages": [
                    {"role": "user", "content": "Which direction does the sun rise from?"},
                    {"role": "assistant", "content": "The sun rises from the east."}
                ],
                "evaluation_keywords": ["east"]
            }
        ]
        
        # Simple instruction following
        instructions = [
            {
                "messages": [
                    {"role": "user", "content": "Please say 'okay'"},
                    {"role": "assistant", "content": "Okay"}
                ],
                "evaluation_keywords": ["okay"]
            },
            {
                "messages": [
                    {"role": "user", "content": "Please repeat: The weather is nice today"},
                    {"role": "assistant", "content": "The weather is nice today"}
                ],
                "evaluation_keywords": ["weather", "nice", "today"]
            },
            {
                "messages": [
                    {"role": "user", "content": "Make a sentence with 'because...so...'"},
                    {"role": "assistant", "content": "Because it rained, so the ground is wet."}
                ],
                "evaluation_keywords": ["because", "so"]
            }
        ]
        
        # Simple conversation
        conversations = [
            {
                "messages": [
                    {"role": "user", "content": "What's your favorite color?"},
                    {"role": "assistant", "content": "As an AI, I don't have personal preferences, but I can help you learn about different colors."}
                ],
                "evaluation_keywords": ["AI", "preferences", "colors"]
            },
            {
                "messages": [
                    {"role": "user", "content": "How are you today?"},
                    {"role": "assistant", "content": "I'm doing well, thank you for asking! Is there anything I can help you with?"}
                ],
                "evaluation_keywords": ["well", "help"]
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