# utils/cleaner.py
import re

def clean_output(text: str) -> str:
    if not text:
        return ""
    patterns = [
        r'<think>.*?</think>',
        r'\*\*think\*\*.*?\*\*end think\*\*',
        r'\*think\*.*?\*end think\*',
        r'\[thinking\].*?\[/thinking\]',
        r'<thinking>.*?</thinking>',
        r'additional_kwargs=\{.*?\}',
        r'response_metadata=\{.*?\}',
        r'AIMessage\(content=.*?\)',
        r'\*\*Answer:\*\*.*?($|\n\n)',
        r'Answer:.*?($|\n\n)',
        r'I should (provide|ask|generate).*?\n',
        r'(Let me|I will|Here\'s a).*?\n',
        r'The question should be.*?example:',
        r'As an AI.*?\n',
        r'I\'m an AI.*?\n',
        r'\*\*Question:\*\*\s*',
        r'\*\*Answer:\*\*.*$'
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()
