# 🤖 RAG AI Interviewer

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-Framework-brightgreen)
![Ollama](https://img.shields.io/badge/Ollama-LLM-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A smart **AI-powered interview simulation system** built with **LangChain**, **RAG (Retrieval-Augmented Generation)**, and **Ollama LLMs**. This tool conducts mock interviews, asks topic-specific questions, and gives feedback — simulating a real interview experience.

---

## 🚀 Features

- 🧠 LLM-based **question generation**
- 📚 **Feedback generation** based on your responses
- 🔁 Prevents **repetitive or too-similar questions**
- 💬 Keeps conversational **context during interviews**
- 🗂️ Uses **Chroma** for RAG-style vector search
- 🧼 Clears memory automatically after each session or when typing `exit`
- 🔧 Modular design with service layers for easy customization

---

## 📁 Project Structure

RAG_AI_Interviewer/ ├── main.py # Main CLI interface ├── config.py # Topic and subtopic mappings ├── requirements.txt # Project dependencies ├── README.md # You're reading this ├── services/ │ ├── interview.py # Interview session logic │ ├── generator.py # Question generator using LangChain │ └── feedback.py # Feedback generator logic ├── utils/ │ ├── logger.py # Custom logging utility │ └── helpers.py # Similarity and helper methods └── docs/ └── chroma/ # Optional Chroma vector DB storage


---

## 📦 Installation

### 🧰 Prerequisites

- Python 3.10 or later
- `pip` or `pipenv`
- [Ollama installed](https://ollama.com/download) and running locally

### 📥 Setup

```bash
git clone https://github.com/your-username/RAG_AI_Interviewer.git
cd RAG_AI_Interviewer

# Optional: create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

💡 Usage
Run the main program:

bash
Copy
Edit
python main.py
Select a topic (e.g., Data Science, Machine Learning)

Enter subtopics (comma-separated, or leave empty for general)

Input the number of questions (default is 3)

Introduce yourself

Answer each question

Type exit to end early — memory will be cleared automatically

Receive feedback at the end 🎯

📚 Configuration
🔹 Topics
You can edit config.py to add or remove topics and subtopics:

python
Copy
Edit
topic_mapping = {
    "data science": ["statistics", "data wrangling", "pandas"],
    "machine learning": ["supervised", "unsupervised", "deep learning"],
    "soft skills": ["communication", "teamwork"],
}

🧠 Memory Management
Uses ConversationBufferMemory from LangChain

Memory is cleared after each session

Typing exit also clears memory immediately

🛠️ Dependencies
See requirements.txt:

txt
Copy
Edit
langchain>=0.2.0
langchain-community>=0.0.21
langchain-core>=0.2.0
langchain-ollama>=0.1.0
chromadb>=0.4.24
tqdm
🧪 Sample Output
bash
Copy
Edit
1. Data Science
2. Machine Learning
3. Soft Skills

Enter topic number: 1
Enter subtopics (comma-separated): pandas, statistics
Number of questions (default 3): 2
Please introduce yourself: I'm a junior data scientist...

[Q1] What are some common data preprocessing techniques used in pandas?
Your Answer: You can use fillna, dropna, etc.

[Q2] How do you handle missing data in pandas?
Your Answer: I use isnull combined with fillna...

Interview Feedback:
- Great job using specific functions like fillna!
- Try to elaborate more on when you'd use each method.

👤 Author

- **Mahmoud Salama**  
  📧 mahmoudsalamacs@gmail.com  
  🔗 [LinkedIn](https://www.linkedin.com/in/mahmoud-salama-5a0525227/)

- **Mennatullah Yasser**  
  🔗 [LinkedIn](https://www.linkedin.com/in/mennatullahyasser12)

- **Talal Ahmed**  
  🔗 [LinkedIn](https://www.linkedin.com/in/talal-ahmed-579905262)


🔗 References
LangChain Docs

Ollama Models

Chroma Vector DB

💡 This project was created as part of my AI and ML portfolio — simulating real interviews with intelligent language models.
