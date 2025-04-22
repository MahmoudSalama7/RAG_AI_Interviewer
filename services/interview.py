# services/interview.py
from langchain.memory import ConversationBufferMemory
from services.generator import QuestionGenerator
from services.feedback import FeedbackGenerator
from utils.logger import get_logger
from utils.helpers import is_similar_question
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module='langchain')


logger = get_logger()

class InterviewSession:
    def __init__(self, model, embeddings, vectordb=None):
        self.model = model
        self.memory = ConversationBufferMemory(return_messages=True)
        self.generator = QuestionGenerator(model, vectordb)
        self.feedbacker = FeedbackGenerator(model)
        self.previous_questions = []

    def conduct_interview(self, topic, subtopics, num_questions):
        for i in range(num_questions):
            subtopic = subtopics[i % len(subtopics)]
            question = self.generator.generate_question(topic, subtopic, self.previous_questions)
            if is_similar_question(question, self.previous_questions):
                logger.info(f"[{i+1}] Skipping too-similar question.")
                continue

            self.previous_questions.append(question)
            logger.info(f"[Q{i+1}] {question}")
            user_answer = input("Your Answer: ").strip()
            if user_answer.lower() == "exit":
                break
            self.memory.chat_memory.add_user_message(user_answer)

    def get_feedback(self, topic):
        history = "\n".join([m.content for m in self.memory.chat_memory.messages])
        feedback = self.feedbacker.generate(topic, history)
        logger.info("\nInterview Feedback:\n" + feedback)

    def clear_memory(self):
        logger.info("Clearing memory after interview session.")
        self.memory.clear()
