# main.py
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.vectorstores import Chroma
from services.interview import InterviewSession
from config import topic_mapping
from utils.logger import get_logger
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module='langchain')


logger = get_logger()

def main():
    logger.info("Initializing model and embeddings...")
    try:
        embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
        model = OllamaLLM(model="deepseek-r1:1.5b", temperature=0.7)
    except:
        embeddings = OllamaEmbeddings()
        model = OllamaLLM(temperature=0.7)
        logger.warning("Fallback to default model")

    vectordb = None
    try:
        vectordb = Chroma(persist_directory='docs/chroma/', embedding_function=embeddings)
    except:
        logger.warning("Failed to load vector DB. Running without it.")

    topics = list(topic_mapping.keys())
    for idx, t in enumerate(topics, 1):
        logger.info(f"{idx}. {t.title()}")

    while True:
        try:
            selected = int(input("Enter topic number: "))
            if 1 <= selected <= len(topics):
                topic = topics[selected - 1]
                break
        except:
            logger.error("Invalid input. Try again.")

    subtopics_input = input("Enter subtopics (comma-separated): ")
    subtopics = [s.strip() for s in subtopics_input.split(",") if s.strip()]
    if not subtopics:
        subtopics = ["general"]

    try:
        num = input("Number of questions (default 3): ")
        num_questions = int(num) if num.strip() else 3
    except:
        num_questions = 3

    candidate_info = input("Please introduce yourself: ")
    logger.info(f"Candidate: {candidate_info}")

    interview = InterviewSession(model, embeddings, vectordb)
    interview.conduct_interview(topic, subtopics, num_questions)
    interview.get_feedback(topic)

    interview.memory.clear()


if __name__ == "__main__":
    main()
