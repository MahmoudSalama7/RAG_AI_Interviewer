# services/feedback.py
from utils.cleaner import clean_output

class FeedbackGenerator:
    def __init__(self, model):
        self.model = model

    def generate(self, topic, chat_history):
        prompt = f"""
You're a technical interviewer reviewing this transcript on {topic}.
Provide a direct, human-style evaluation with strengths, weaknesses, and study suggestions.

Transcript:
{chat_history}
"""
        try:
            return clean_output(self.model.invoke(prompt))
        except Exception:
            return "Your interview responses demonstrated several areas of technical competence."
