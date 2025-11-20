# 📚 RAG Document QA System

A production-ready Retrieval-Augmented Generation (RAG) system with **two interfaces**: Legacy Streamlit and Modern FastAPI + React full-stack application.

## 🌟 Features

- **📤 Dynamic Document Upload** - Upload PDFs, DOCX, TXT, MD files on-the-fly
- **🤖 AI-Powered Q&A** - OpenAI integration for intelligent answers
- **🔍 Semantic Search** - Elasticsearch vector storage with BGE embeddings
- **💬 Modern Chat UI** - React-based interface with real-time updates
- **🔢 Token Tracking** - Monitor usage with detailed breakdown
- **📚 Source Attribution** - See which documents were used
- **🎨 Two UIs** - Choose Streamlit (simple) or FastAPI+React (production)

## 🏗️ Architecture

```
┌─────────────┐      HTTP/REST      ┌──────────────┐
│   React     │ ◄──────────────────►│   FastAPI    │
│  Frontend   │                     │   Backend    │
│  (Port 3000)│                     │  (Port 8000) │
└─────────────┘                     └──────────────┘
                                           │
                                           ▼
                                    ┌──────────────┐
                                    │Elasticsearch │
                                    │  (Port 9200) │
                                    └──────────────┘
                                           │
                                           ▼
                                    ┌───────────────┐
                                    │   OpenAI      │
                                    │  gpt-4.1-nano │
                                    └───────────────┘
```

## 📁 Project Structure

```
sampleprojects/
├── backend/              # FastAPI backend
│   └── api.py           # RESTful API
├── frontend/            # React frontend
│   ├── src/
│   │   ├── components/  # UI components
│   │   ├── services/    # API integration
│   │   └── App.js       # Main app
│   └── package.json
├── src/rag_qa/          # Core RAG system
│   ├── core/
│   │   ├── rag_openai.py       # OpenAI integration
│   │   ├── es7_retriever.py    # Elasticsearch retrieval
│   │   └── document_processor.py
│   └── utils/
├── config/
│   └── config.yml       # Unified configuration
├── requirements.txt     # All Python dependencies
├── package.json         # Frontend dependencies
├── app.py              # Streamlit UI (legacy)
└── main.py             # Terminal interface
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+ and npm
- Docker (for Elasticsearch)

### Installation

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install Frontend dependencies
npm install

# 3. Start Elasticsearch
docker-compose up -d

# 4. Configure API key
# Edit config/config.yml and add your OpenAI API key
```

### Run the Application

**Option 1: Full-Stack Web UI (Recommended)**

```bash
# Terminal 1 - Backend
cd backend
python3 api.py

# Terminal 2 - Frontend
npm start

# Open http://localhost:3000
```

**Option 2: Terminal Interface**

```bash
python3 main.py
```

## 🔧 Configuration

Edit `config/config.yml`:

```yaml
llm:
  use_openai: true
  api_key: "your-openai-api-key"
  model: "gpt-4.1-nano"  # or gpt-4o-mini for cheaper
  params:
    max_tokens: 1000
    temperature: 0

vector_store:
  elasticsearch:
    es_url: "http://localhost:9200"
    index_name: "rag_pdf_chunks_v1"

retrieval:
  top_k: 3  # Number of chunks to retrieve
```

## 📖 Documentation

- **[INSTALL.md](INSTALL.md)** - Quick installation guide
- **[README_FULLSTACK.md](README_FULLSTACK.md)** - Full-stack application guide
- **[FASTAPI_REACT_SETUP.md](FASTAPI_REACT_SETUP.md)** - Detailed setup & troubleshooting

## 🎯 Key Features

- ✅ **Dynamic Document Upload** - Upload files on-the-fly
- ✅ **Advanced Chat Interface** - Modern React UI
- ✅ **Detailed Token Tracking** - Monitor usage
- ✅ **Source Attribution** - Expandable sources
- ✅ **Multi-Session Support** - Per-user isolation
- ✅ **RESTful API** - Full API access
- ✅ **Production Ready** - Scalable architecture

## 🔧 Tech Stack

**Backend:**
- FastAPI - Modern Python web framework
- LangChain - RAG pipeline orchestration
- Elasticsearch - Vector storage & retrieval
- OpenAI - Language model (gpt-4.1-nano, gpt-4o-mini)
- Sentence Transformers - BGE embeddings (768-dim)

**Frontend:**
- React 18 - UI framework
- Axios - HTTP client
- react-dropzone - File uploads
- react-markdown - Markdown rendering

## 🎨 Screenshots

**Modern React UI:**
- Clean sidebar with document management
- Real-time chat interface
- Token usage display
- Source attribution

**Streamlit UI:**
- Simple chat interface
- Pre-indexed documents
- Basic Q&A functionality

## 📊 API Endpoints

```
POST   /api/session/create        - Create new session
POST   /api/documents/upload      - Upload document
POST   /api/query                 - Ask question
GET    /api/session/{id}          - Get session info
DELETE /api/session/{id}          - Delete session
```

**API Docs:** http://localhost:8000/docs

## 🔍 How It Works

### 1. Document Ingestion
- Upload PDF/DOCX/TXT/MD files
- Process with UnstructuredFileIOLoader
- Split into chunks (tiktoken-based, 100 tokens)
- Generate 768-dim BGE embeddings
- Store in Elasticsearch

### 2. Query Processing
- User asks a question
- Generate query embedding
- Retrieve top-K similar chunks (cosine similarity)
- Build context from retrieved chunks

### 3. Answer Generation
- Send context + question to OpenAI
- Generate concise answer (max 50 words)
- Track token usage (prompt + completion)
- Display sources used

## 🐛 Troubleshooting

**Elasticsearch not running:**
```bash
docker-compose up -d
curl http://localhost:9200
```

**Module not found:**
```bash
pip install -r requirements.txt
```

**Port already in use:**
```bash
lsof -ti:8000 | xargs kill -9  # Backend
lsof -ti:3000 | xargs kill -9  # Frontend
```

**node_modules being committed:**
- Already added to `.gitignore`
- Run: `git rm -r --cached node_modules`

## 🚀 Production Deployment

**Backend:**
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend.api:app
```

**Frontend:**
```bash
cd frontend && npm run build
# Serve build/ with nginx
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open pull request

## 📄 License

MIT License - See LICENSE file

## 🙏 Credits

- **OpenAI** - Language models
- **Elasticsearch** - Vector storage
- **LangChain** - RAG framework
- **React** - Frontend framework
- **FastAPI** - Backend framework

---

**Made with ❤️ using FastAPI + React + OpenAI**
