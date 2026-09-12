# RockyBot: News Research Tool 📈

**RockyBot** is a Retrieval-Augmented Generation (RAG) based news research application that allows users to provide multiple news article URLs and ask questions about their content.

The application loads and processes articles, splits them into smaller chunks, converts the chunks into semantic embeddings using a Sentence Transformer model, stores them in a FAISS vector index, retrieves the most relevant chunks for a user query, and uses **Google Gemini** to generate an answer based on the retrieved context.

## 🚀 Live Demo

**[Try RockyBot Live](https://sumifyer.streamlit.app/)**

## ✨ Features

* Load content from multiple news article URLs
* Process and clean article content
* Split documents into smaller chunks
* Generate semantic embeddings using Sentence Transformers
* Store embeddings using FAISS
* Perform similarity-based retrieval
* Retrieve the most relevant chunks for a user query
* Generate answers using Google Gemini
* Display retrieved chunks for transparency
* Display source URLs
* Interactive Streamlit web interface

## 🧠 How It Works

RockyBot follows a Retrieval-Augmented Generation pipeline:

```text
News Article URLs
        ↓
Document Loading
        ↓
Text Processing
        ↓
Text Splitting
        ↓
Text Chunks
        ↓
Sentence Transformer
        ↓
Embeddings
        ↓
FAISS Vector Index
        ↓
User Question
        ↓
Query Embedding
        ↓
Similarity Search
        ↓
Top-K Relevant Chunks
        ↓
Retrieved Context
        ↓
Google Gemini
        ↓
Generated Answer
```

## 🛠️ Technologies Used

* **Python**
* **Streamlit**
* **LangChain**
* **Sentence Transformers**
* **FAISS**
* **Google Gemini API**
* **Unstructured**
* **spaCy**
* **BeautifulSoup**
* **Requests**
* **python-dotenv**

## 📂 Project Structure

```text
text-summarizer/
│
├── final-txt-summarize-rag-pro.py
├── requirements.txt
├── README.md
├── .gitignore
└── ...
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/ursuva/text-summarizer.git
cd text-summarizer
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 🔑 API Key Setup

RockyBot uses the **Google Gemini API** for answer generation.

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_api_key_here
```

Do not commit your `.env` file to GitHub.

Add the following to `.gitignore`:

```text
.env
__pycache__/
```

For Streamlit Community Cloud, add your API key through the application's **Secrets** configuration rather than committing it to the repository.

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run final-txt-summarize-rag-pro.py
```

The application will open in your browser.

## 📖 Usage

### Step 1: Add News URLs

Enter news article URLs in the application.

RockyBot can process multiple URLs and combine their content into a searchable knowledge base.

### Step 2: Process URLs

Click **Process URLs**.

The application:

1. Loads the articles
2. Extracts and processes the text
3. Splits the documents into chunks
4. Generates embeddings
5. Builds the FAISS vector index

### Step 3: Ask a Question

Enter a question related to the processed articles.

Example:

```text
What happened to Tata Motors?
```

### Step 4: Retrieve Relevant Information

The question is converted into an embedding.

FAISS performs a similarity search against the stored document embeddings and retrieves the most relevant chunks.

### Step 5: Generate the Answer

The retrieved chunks are provided to **Google Gemini** as context.

Gemini then generates the final answer based on the retrieved information.

The application also displays the retrieved chunks and source URLs to provide transparency into the answer generation process.

## 🔍 RAG Components

### 1. Document Loading

News articles are loaded from the provided URLs using document-loading and web-processing components.

### 2. Text Splitting

Large documents are divided into smaller chunks using:

```text
RecursiveCharacterTextSplitter
```

Chunking makes the documents easier to embed and retrieve efficiently.

### 3. Embeddings

Each chunk is converted into a numerical vector using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

These vectors represent the semantic meaning of the text.

### 4. FAISS Vector Store

FAISS stores the generated embeddings and performs efficient similarity searches.

The project uses:

```text
faiss.IndexFlatL2
```

### 5. Retrieval

When a user asks a question, the question is converted into an embedding using the same embedding model.

FAISS compares the query vector with the stored document vectors and retrieves the most relevant chunks.

### 6. Generation

The retrieved chunks are passed to Google Gemini as context.

Gemini uses this retrieved context to generate the final response.

## 🎯 Key Learning Outcomes

Through this project, I explored and implemented:

* Retrieval-Augmented Generation (RAG)
* Text preprocessing and document cleaning
* Text chunking
* Semantic embeddings
* Sentence Transformers
* Vector similarity search
* FAISS
* LangChain
* Prompt construction
* Large Language Models
* Google Gemini API
* Streamlit application development
* RAG-based question answering
* Cloud deployment

## 🚀 Future Improvements

* Replace deprecated URL-loading components with modern alternatives
* Add support for additional news sources
* Add persistent vector database storage
* Improve document cleaning and preprocessing
* Add conversation history
* Add metadata-based filtering
* Add Docker support
* Improve retrieval and ranking strategies
* Add evaluation metrics for RAG responses

## ☁️ Deployment

RockyBot is deployed using **Streamlit Community Cloud**.

## 👨‍💻 Author

**Suvajit Biswas**

Computer Science graduate specializing in IoT, Cyber Security, and Blockchain Technology.

GitHub: https://github.com/ursuva


