# 🎬 YouTube RAG Chat — AI-Powered Video Q&A System

> Chat with any YouTube video in any language using Retrieval-Augmented Generation (RAG)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-latest-green.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)
![FAISS](https://img.shields.io/badge/FAISS-Vector--Store-purple.svg)

---

## 🚀 Project Overview

This project builds an end-to-end **RAG (Retrieval-Augmented Generation)** system that allows users to have intelligent conversations with any YouTube video. Simply paste a YouTube URL and ask questions — the AI answers based strictly on the video content.

The system supports **multilingual videos** (Hindi, Spanish, French, Tamil, etc.) by automatically detecting and translating transcripts to English before processing.

---

## 🏗️ System Architecture---

## ✨ Features

- 🎯 **RAG Pipeline** — Answers grounded strictly in video content, no hallucinations
- 🌐 **Multilingual Support** — Auto-detects and translates Hindi, Spanish, French, Tamil and 100+ languages
- 🔍 **MMR Retrieval** — Maximum Marginal Relevance for diverse and precise chunk retrieval
- 💬 **Chat Interface** — Full conversational UI with chat history
- 🧪 **RAGAS Evaluation** — Measures Faithfulness, Answer Relevancy, Context Precision and Recall
- 📊 **LangSmith Tracing** — Full pipeline observability and debugging
- 🔌 **Chrome Extension** — Chat with videos directly on YouTube
- 🌍 **Streamlit Web App** — Clean web UI to load any video and ask questions
- ⚡ **FastAPI Backend** — REST API for the Chrome Extension

---

## 📁 Project Structure---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | FAISS |
| RAG Framework | LangChain |
| Translation | GoogleTranslator / googletrans |
| Transcript | youtube-transcript-api |
| Web UI | Streamlit |
| API | FastAPI + Uvicorn |
| Browser Extension | Chrome Extension (Manifest V3) |
| Evaluation | RAGAS + LangSmith |
| Tunneling | ngrok |

---

## 📊 RAGAS Evaluation Results

Evaluated on the DeepMind/Demis Hassabis interview video:

| Metric | Score | Description |
|--------|-------|-------------|
| **Faithfulness** | 0.89 | Answers grounded in video content |
| **Answer Relevancy** | 0.96 | Answers relevant to questions |
| **Context Precision** | 0.68 | Relevant chunks retrieved |
| **Context Recall** | 0.80 | All relevant info captured |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- OpenAI API Key
- Google Chrome (for extension)

### 1. Clone the repository
```bash
git clone https://github.com/khusbu123q/YouTube-rag-project.git
cd YouTube-rag-project
```

### 2. Install dependencies
```bash
cd streamlit_app
pip install -r requirements.txt
```

### 3. Set up environment variables
```bash
# Create .env file
echo "OPENAI_API_KEY=your-openai-api-key-here" > .env
```

### 4. Run Streamlit App
```bash
streamlit run app.py
```

### 5. Run FastAPI Backend (for Chrome Extension)
```bash
uvicorn backend:app --reload --port 8000
```

---

## 🔌 Chrome Extension Setup

1. Open Chrome → `chrome://extensions`
2. Enable **Developer Mode**
3. Click **Load Unpacked**
4. Select the `extension/` folder
5. Open any YouTube video
6. Click the extension icon and start chatting!

---

## 📓 Colab Notebooks

The `colab_notebooks/` folder contains the full research pipeline:

- ✅ YouTube transcript fetching with multilingual support
- ✅ Text splitting and chunking strategies
- ✅ Embedding generation and FAISS vector store
- ✅ RAG chain with MMR retrieval
- ✅ RAGAS evaluation with Faithfulness, Relevancy, Precision, Recall
- ✅ LangSmith tracing and observability

---

## 🎯 How It Works

1. **User pastes a YouTube URL** in the Streamlit sidebar
2. **Transcript is fetched** using youtube-transcript-api
3. **If non-English**, transcript is auto-translated to English
4. **Text is split** into 500-character chunks with 100-character overlap
5. **Embeddings are generated** using OpenAI text-embedding-3-small
6. **Chunks stored** in FAISS vector store
7. **User asks a question** in the chat interface
8. **MMR retriever** finds the 5 most relevant chunks
9. **GPT-4o-mini** generates an answer strictly from the retrieved chunks
10. **Answer displayed** in the chat UI

---

## 👩‍💻 Author

**Khusbu Agarwal**
- GitHub: [@khusbu123q](https://github.com/khusbu123q)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
