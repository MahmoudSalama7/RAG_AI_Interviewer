# services/generator.py
from config import topic_mapping
from utils.cleaner import clean_output

class QuestionGenerator:
    def __init__(self, model, vectordb=None):
        self.model = model
        self.vectordb = vectordb

    def generate_question(self, topic, subtopic, previous_questions=None):
        if previous_questions is None:
            previous_questions = []
        context = ""
        if self.vectordb:
            try:
                filter_dict = {"topic": topic_mapping.get(topic, topic)}
                docs = self.vectordb.similarity_search(f"{topic} {subtopic} interview questions", filter=filter_dict, k=3)
                context = "\n\n".join([doc.page_content for doc in docs])[:1500]
            except Exception:
                pass

        prev_qs_text = "\n".join([f"- {q}" for q in previous_questions]) or "None"
        prompt = f"""
You are an expert technical interviewer conducting an interview on {topic}, subtopic: {subtopic}.
Avoid repeating these questions:
{prev_qs_text}
{context}

Generate one technical, real-world question with depth. No AI disclaimers or explanations.
"""
        try:
            result = self.model.invoke(prompt)
            return clean_output(result)
        except Exception:
            return f"What are the challenges you've faced with {subtopic} in {topic}?"
