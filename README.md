# Text Summarizer & RAG Application

An AI-powered text summarization and Retrieval-Augmented Generation (RAG) application built with Python and Streamlit.

## 🚀 Live Demo

[Open the Live App](https://sumifyer.streamlit.app/)

## 📌 Features

* Summarize text and web content
* Extract and process content from URLs
* Text splitting and chunking
* Generate semantic embeddings
* FAISS-based similarity search
* Retrieve relevant context for user queries
* AI-generated responses using Google Gemini
* Interactive Streamlit interface
* Supports web-based content processing

## 🛠️ Tech Stack

* **Python**
* **Streamlit**
* **LangChain**
* **FAISS**
* **Sentence Transformers**
* **Google Gemini**
* **Unstructured**
* **BeautifulSoup**
* **spaCy**
* **Requests**

## 🔄 RAG Workflow

```text
Input Text / URL
       ↓
Document Loading
       ↓
Text Cleaning
       ↓
Text Splitting
       ↓
Chunk Creation
       ↓
Embedding Generation
       ↓
FAISS Vector Store
       ↓
Similarity Search
       ↓
Relevant Context
       ↓
Google Gemini
       ↓
Generated Response / Summary
```

## 📂 Project Structure

```text
text-summarizer/
│
├── final-txt-summarize-rag-pro.py
├── requirements.txt
├── README.md
└── ...
```

## ⚙️ Run Locally

Clone the repository:

```bash
git clone https://github.com/ursuva/text-summarizer.git
cd text-summarizer
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file and add your API key:

```env
GOOGLE_API_KEY=your_api_key_here
```

Run the application:

```bash
streamlit run final-txt-summarize-rag-pro.py
```

The application will be available at:

```text
http://localhost:8501
```

## ☁️ Deployment

The application is deployed using Streamlit Community Cloud.

## 📚 What I Learned

This project helped me understand and implement:

* Document loaders
* Text chunking
* Embeddings
* Vector databases
* FAISS similarity search
* Retrieval-Augmented Generation
* LLM prompting
* Streamlit application development
* Dependency management
* Cloud deployment

## 👨‍💻 Author

**Suvajit Biswas**

GitHub: https://github.com/ursuva
