"""
Math tool-use datasets for calculator warmup SFT.

The JSONL format is one conversation object per line:
{"messages": [{"role": "user", ...}, {"role": "assistant", "content": [...]}], "meta": {...}}

Assistant content may be a list of parts using nanochat's native tool schema:
- {"type": "text", "text": "..."}
- {"type": "python", "text": "18*7"}
- {"type": "python_output", "text": "126"}
"""

import json
import os

from tasks.common import Task


class MathToolJSON(Task):
    """Load structure-preserving math tool-use JSONL conversations."""

    def __init__(self, filepath, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        self.conversations = []

        assert os.path.exists(filepath), f"Math tool data not found: {filepath}"
        with open(filepath, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                messages = row["messages"] if isinstance(row, dict) else row
                assert isinstance(messages, list), f"Line {line_idx}: messages must be a list"
                assert len(messages) >= 2, f"Line {line_idx}: conversation must have at least 2 messages"
                for i, message in enumerate(messages):
                    expected_role = "user" if i % 2 == 0 else "assistant"
                    assert message["role"] == expected_role, f"Line {line_idx}: message {i} role mismatch"
                    content = message["content"]
                    if expected_role == "user":
                        assert isinstance(content, str), f"Line {line_idx}: user content must be str"
                    else:
                        assert isinstance(content, (str, list)), f"Line {line_idx}: assistant content must be str|list"
                        if isinstance(content, list):
                            for part in content:
                                assert part["type"] in {"text", "python", "python_output"}
                                assert isinstance(part["text"], str)
                self.conversations.append({"messages": messages})

    def num_examples(self):
        return len(self.conversations)

    def get_example(self, index):
        return self.conversations[index]
