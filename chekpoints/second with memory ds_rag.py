from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.vectorstores import Chroma
import random
import re
import os
import time
import warnings
import sys

# Suppress warning messages
warnings.filterwarnings('ignore')

# Topic mapping between user-friendly names and metadata values
topic_mapping = {
    "machine learning": "machine_learning",
    "cyber security": "cybersecurity",
    "data engineering": "data_engineering",
    "devops": "devops"
}

# Clean output - remove thinking markers and other artifacts
def clean_output(text: str) -> str:
    """Remove thinking markers, meta-instructions and other artifacts from output"""
    if not text:
        return ""
        
    # Remove "thinking" sections with various formats
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'\*\*think\*\*.*?\*\*end think\*\*', '', text, flags=re.DOTALL)
    text = re.sub(r'\*think\*.*?\*end think\*', '', text, flags=re.DOTALL)
    text = re.sub(r'\[thinking\].*?\[/thinking\]', '', text, flags=re.DOTALL)
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
    
    # Remove metadata and kwargs
    text = re.sub(r'additional_kwargs=\{.*?\}', '', text, flags=re.DOTALL)
    text = re.sub(r'response_metadata=\{.*?\}', '', text, flags=re.DOTALL)
    text = re.sub(r'AIMessage\(content=.*?\)', '', text, flags=re.DOTALL)
    
    # Remove answers and explanations
    text = re.sub(r'\*\*Answer:\*\*.*?($|\n\n)', '', text, flags=re.DOTALL)
    text = re.sub(r'Answer:.*?($|\n\n)', '', text, flags=re.DOTALL)
    
    # Remove meta-instructions like "I should..."
    text = re.sub(r'I should (provide|ask|generate).*?\n', '', text)
    
    # Remove any "Let me..." or "I will..." statements
    text = re.sub(r'(Let me|I will|Here\'s a).*?\n', '', text)
    text = re.sub(r'The question should be.*?example:', '', text)
    
    # Remove empty lines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Remove phrases that reveal it's an AI
    text = re.sub(r'As an AI.*?\n', '', text)
    text = re.sub(r'I\'m an AI.*?\n', '', text)
    
    # Remove **Question:** prefix if present
    text = re.sub(r'\*\*Question:\*\*\s*', '', text)
    
    # Remove **Answer:** sections if present
    text = re.sub(r'\*\*Answer:\*\*.*$', '', text, flags=re.DOTALL)
    
    return text.strip()

def print_separator(symbol="=", length=50):
    """Print a separator line with given symbol and length"""
    print("\n" + symbol*length)

def print_header(text):
    """Print a header with separators"""
    print_separator()
    print(text)
    print_separator()

def is_similar_question(new_question, existing_questions, similarity_threshold=0.7):
    """Check if a new question is too similar to existing ones"""
    if not existing_questions:
        return False
        
    new_question = new_question.lower()
    
    for existing in existing_questions:
        existing = existing.lower()
        
        # Basic similarity check
        if len(new_question) > 0 and len(existing) > 0:
            # Jaccard similarity for word overlap
            new_words = set(new_question.split())
            existing_words = set(existing.split())
            
            if not new_words or not existing_words:
                continue
                
            intersection = len(new_words.intersection(existing_words))
            union = len(new_words.union(existing_words))
            
            if union > 0 and intersection / union > similarity_threshold:
                return True
                
    return False

# Simplified question generation with better context
def generate_interview_question(model, topic, subtopic, previous_questions=None, vectordb=None):
    """Generate an interview question using direct LLM call with improved context"""
    if previous_questions is None:
        previous_questions = []
    
    previous_questions_text = "\n".join([f"- {q}" for q in previous_questions]) if previous_questions else "None"
    
    # Try to get some context from vectordb if available
    context = ""
    if vectordb:
        try:
            # Get relevant documents for context
            filter_dict = {"topic": topic_mapping.get(topic, topic)}
            search_query = f"{topic} {subtopic} interview questions"
            docs = vectordb.similarity_search(search_query, filter=filter_dict, k=3)
            
            # Extract content from documents
            context_texts = [doc.page_content for doc in docs]
            context = "\n\n".join(context_texts)
            
            # Limit context length
            if len(context) > 1500:
                context = context[:1500] + "..."
        except:
            # If retrieval fails, continue with empty context
            pass
    
    prompt = f"""
You are an expert technical interviewer conducting an interview on {topic}. 
Focus specifically on this subtopic: {subtopic}.

Generate a challenging interview question about {subtopic} in {topic}.
The question should test the candidate's technical knowledge.

IMPORTANT: Your question MUST be different from these previous questions:
{previous_questions_text}

{context}

IMPORTANT:
1. Generate your response as if you are a human interviewer.
2. Don't include any phrases revealing you're an AI.
3. Just provide the direct interview question as a human interviewer would ask it.
4. Don't include any explanation or answer to the question you generate.
5. Make the question realistic, challenging, and relevant to real-world scenarios.
6. The question should require technical depth to answer properly.
"""
    try:
        # Try to generate a question with the current model
        result = model.invoke(prompt)
        question = clean_output(result)
        
        # Check if question is valid
        if not question or len(question) < 10 or question.lower().startswith("i ") or question.lower().startswith("as an"):
            # Try one more time with higher temperature
            model.temperature = 0.9
            result = model.invoke(prompt)
            question = clean_output(result)
            model.temperature = 0.7
        
        return question
    except Exception as e:
        # Simple fallback for errors
        return f"What are the key challenges you've faced when working with {subtopic} in {topic}?"

# Improved feedback generation
def generate_feedback(model, topic, chat_history):
    """Generate comprehensive feedback based on the interview"""
    
    feedback_prompt = f"""
You are a technical interviewer who has just finished interviewing a candidate on {topic}.
Review the interview transcript and provide comprehensive feedback.

Full interview transcript:
{chat_history}

Provide a direct, detailed assessment of the candidate's performance, including:
1. Overall strengths and areas for improvement
2. Technical knowledge assessment
3. Specific concepts the candidate demonstrated strong understanding of
4. Specific concepts where the candidate could improve
5. Recommendations for further study (if applicable)

Format your feedback as if you are a human interviewer giving direct feedback.
Don't include any phrases revealing you're an AI assistant.
"""
    
    try:
        feedback_response = model.invoke(feedback_prompt)
        return clean_output(feedback_response)
    except Exception as e:
        # Simple fallback
        return "Thank you for completing the interview. Your responses showed technical knowledge in several areas."

def run_interview():
    """Main function to run the technical interview simulation"""
    
    print_header("Welcome to the Technical Interview Simulator")
    
    try:
        # Load embeddings & model
        print("Initializing AI interviewer...", end="", flush=True)
        try:
            embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
            model = OllamaLLM(model="deepseek-r1:1.5b", temperature=0.7)
            print(" Done!")
        except Exception as e:
            # Try with default model as fallback
            embeddings = OllamaEmbeddings()
            model = OllamaLLM(temperature=0.7)
            print(" Done! (using default model)")
    except Exception as e:
        print("\nError: Could not initialize the Ollama model.")
        print(f"Error details: {str(e)}")
        return

    vectordb = None
    try:
        # Load vector database
        print("Loading knowledge base...", end="", flush=True)
        persist_directory = 'docs/chroma/'
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        print(" Done!")
    except Exception as e:
        print(" Failed to load vector database. Continuing without it.")
        # Continue without vectordb - direct generation will be used

    # Simple memory to store the conversation
    memory = ConversationBufferMemory(return_messages=True)

    # Topic Selection
    topics = list(topic_mapping.keys())
    print("\nChoose a topic for your interview:")
    for idx, topic in enumerate(topics, 1):
        print(f"{idx}. {topic.capitalize()}")

    while True:
        try:
            choice = int(input("Enter topic number: "))
            if 1 <= choice <= len(topics):
                selected_topic = topics[choice - 1]
                break
            else:
                print("Invalid choice, try again.")
        except ValueError:
            print("Please enter a valid number.")

    print(f"\nYou selected: {selected_topic.capitalize()}")

    # Get subtopics from user
    print("\nEnter specific subtopics you want to focus on (comma-separated).")
    print("For example: 'linear regression, neural networks' for machine learning")
    subtopics_input = input("\nSubtopics: ")
    subtopics = [s.strip() for s in subtopics_input.split(",") if s.strip()]

    if not subtopics:
        print("No subtopics specified. Using general topics.")
        subtopics = ["general knowledge"]

    # Get number of questions
    num_questions = 3  # Default
    try:
        num_input = input("\nHow many questions would you like? (default: 3): ")
        if num_input.strip():
            num_questions = max(1, min(10, int(num_input)))
    except ValueError:
        print("Using default of 3 questions.")

    # Introduction with interview format explanation
    print_header("Interview Setup")
    print(f"This will be a {num_questions}-question technical interview on {selected_topic.capitalize()}")
    print(f"Focusing on: {', '.join(subtopics)}")
    print("\nFormat: I'll ask you questions one by one.")
    print("Type your answers after each question.")
    print("You can type 'exit' at any time to end the interview.")
    
    # Get candidate info
    candidate_info = input("\nPlease introduce yourself (name and background): ")

    # Try to extract name from the info
    name_match = re.search(r'^(?:I am|My name is|I\'m)?\s*([A-Za-z]+)', candidate_info)
    if name_match:
        candidate_name = name_match.group(1)
    else:
        # If name can't be easily extracted, use a generic term
        candidate_name = "Candidate"

    print(f"\nThank you, {candidate_name}. Let's begin your {selected_topic} interview.")
    
    # Create a plan for distributing subtopics across questions
    question_plan = []
    while len(question_plan) < num_questions:
        # Cycle through subtopics until we have enough questions
        question_plan.extend(subtopics)
    question_plan = question_plan[:num_questions]  # Trim to exact number
    random.shuffle(question_plan)  # Randomize order

    # Store all Q&A for final feedback
    all_questions = []
    chat_history = []

    # Interview Loop
    print_header("Interview Starting")
    time.sleep(1)  # Brief pause for effect
    
    question_num = 1
    for current_subtopic in question_plan:
        print(f"\n[Question {question_num}/{num_questions}]")
        
        # Generate a unique question (with up to 3 attempts)
        question = None
        max_attempts = 3
        
        for attempt in range(max_attempts):
            # Generate question
            generated_question = generate_interview_question(
                model, 
                selected_topic, 
                current_subtopic, 
                all_questions,
                vectordb
            )
            
            # Check if it's sufficiently different from previous questions
            if not is_similar_question(generated_question, all_questions):
                question = generated_question
                break
                
            # Last attempt - use anyway
            if attempt == max_attempts - 1:
                question = generated_question
        
        # Display the question
        print(f"\nQ{question_num}: {question}")
        
        # Add to list of asked questions
        all_questions.append(question)

        # Get user's answer
        user_answer = input("Your Answer: ")

        if user_answer.lower() == "exit":
            print("\nInterview Ended.")
            return

        # Store in chat history for final feedback
        chat_history.append(f"Q{question_num}: {question}")
        chat_history.append(f"Answer: {user_answer}")
        
        # Move to next question
        question_num += 1

    # Provide final feedback
    if len(all_questions) > 0:
        print_header("Interview Complete")
        
        # Generate feedback
        final_feedback = generate_feedback(
            model, 
            selected_topic, 
            "\n".join(chat_history)
        )
        
        print_header("Interview Feedback")
        print(final_feedback)

    print_separator()
    print(f"Thank you for participating in the {selected_topic} interview, {candidate_name}.")
    memory.clear()

if __name__ == "__main__":
    run_interview()