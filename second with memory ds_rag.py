from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
import os

# Define the model and embeddings
embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
model = OllamaLLM(model="deepseek-r1:1.5b")

# Load and split documents
# pdf_file = "C:/Users/LENOVO/Downloads/docs/ML Topics for Fine-Tuning  (1).pdf"
# loader = PyPDFLoader(pdf_file)
# pages = loader.load()

content_dir = "/docs"
pdf_files = [f for f in os.listdir(content_dir) if f.endswith('.pdf')]
all_pages = []
for pdf_file in pdf_files:
    file_path = os.path.join(content_dir, pdf_file)
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    all_pages.extend(pages)

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n", "\n\n", " "]
)
docs = text_splitter.split_documents(all_pages)

# Create the vector database
persist_directory = 'docs/chroma/'
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=persist_directory
)

# Initialize Conversation Summary Buffer Memory
memory = ConversationSummaryBufferMemory(
    llm=model,  # Your Ollama LLM
    max_token_limit=1000,  # Limit the memory to 1000 tokens
    return_messages=True  # Return messages as a list
)

# Define the prompt template for generating interview questions
question_prompt_template = """
You are an interviewer conducting a technical interview. Your task is to generate a unique and relevant interview question based on the context below.

Context: {context}

Generate a question that tests the candidate's understanding of the topic. The question should be clear and concise.

Question:
"""
QUESTION_PROMPT = PromptTemplate.from_template(question_prompt_template)

# Define the prompt template for evaluating the user's answer
feedback_prompt_template = """
You are an interviewer evaluating a candidate's response to an interview question. Below is the question, the candidate's answer, and the context.

Question: {question}
Answer: {answer}
Context: {context}

Evaluate the candidate's answer and provide feedback. Highlight what was good and what could be improved. Keep the feedback constructive and professional.

Feedback:
"""
FEEDBACK_PROMPT = PromptTemplate.from_template(feedback_prompt_template)

# Create the RetrievalQA chain for question generation
qa_chain = RetrievalQA.from_chain_type(
    llm=model,  # Your Ollama LLM
    chain_type="stuff",
    retriever=vectordb.as_retriever(search_kwargs={"k": 5}),  # Retrieve top 5 documents
    return_source_documents=True  # Return source documents for reference
)

# Start the conversation loop
print("Welcome to the  interview system! Say 'exit' to end the conversation.")
score = 0

# Interviewer Introduction
print("\nInterviewer: Hello! Welcome to the interview. I'll be asking you some technical questions to assess your understanding of machine learning concepts.")
print("Let's get started!\n")

while True:
    # Retrieve context dynamically (for simplicity, we'll use the first document)
    context = docs[0].page_content[:500]  # Use the first 500 characters of the first document as context

    # Generate a unique interview question based on the context
    question_prompt = QUESTION_PROMPT.format(context=context)
    question_result = model(question_prompt)
    question = question_result.strip()  # Ensure the question is clean

    # Print the generated question
    print(f"Interviewer: {question}")

    # Get user input
    user_answer = input("Your Answer: ")

    # Exit the loop if the user types 'exit'
    if user_answer.lower() == "exit":
        print("Interviewer: Thank you for your time! Goodbye!")
        break

    # Evaluate the user's answer
    feedback_prompt = FEEDBACK_PROMPT.format(question=question, answer=user_answer, context=context)
    feedback_result = model(feedback_prompt)
    feedback = feedback_result.strip()  # Ensure the feedback is clean

    # Print the feedback
    print(f"\nInterviewer Feedback: {feedback}")

    # Calculate and print the score
    if any(kw in feedback.lower() for kw in ['good', 'correct', 'accurate', 'proper', 'excellent']):
        score += 1
    print(f"Current Score: {score}/10")

    # Save the conversation to memory
    memory.save_context(
        {"input": question},  # Save the question as input
        {"output": f"Answer: {user_answer}\nFeedback: {feedback}"}  # Save the answer and feedback as output
    )

    # Retrieve the conversation history from memory
    conversation_history = memory.load_memory_variables({})
    print("\nConversation History:")
    print(conversation_history["history"])