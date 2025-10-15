# 📚 Data-Driven Document QA Using RAG and Vector Embeddings

A comprehensive Retrieval-Augmented Generation (RAG) system that combines document retrieval with language models to provide accurate, context-aware answers based on your documents.

## 🎯 Abstract

This project develops a RAG system using **free resources only**: Hugging Face transformers, LangChain, and FAISS. The system can ingest documents (PDFs, DOCX, TXT, MD), convert them into vector embeddings, and store them in a vector database for fast semantic search. When users ask questions, the system retrieves relevant information and uses a Large Language Model to generate precise, context-aware answers.

Unlike standard chatbots that rely on pre-trained knowledge, this system answers based on provided documents, making it more reliable and reducing hallucinations. The project demonstrates core Data Science techniques: text preprocessing, embeddings, similarity search, and NLP integration.

## 🌟 Features

- **📄 Multi-format Document Support**: PDF, DOCX, TXT, Markdown
- **🧠 Semantic Search**: Vector embeddings using Sentence Transformers
- **🤖 AI-Powered Answers**: Language model integration for answer generation
- **🌐 Web Interface**: User-friendly Streamlit application
- **📊 Analytics & Evaluation**: Performance metrics and system monitoring
- **⚙️ Configurable**: Flexible parameters for different use cases
- **💰 100% Free Resources**: No paid APIs or services required

## 🏗️ Architecture

```
📁 Project Structure
├── src/rag_qa/              # Main package
│   ├── core/                # Core system components
│   │   ├── document_processor.py    # Document loading & processing
│   │   └── rag_system.py           # Main RAG system
│   ├── utils/               # Utility modules
│   │   ├── config.py        # Configuration management
│   │   ├── evaluation.py    # System evaluation
│   │   └── helpers.py       # Helper functions
│   └── ui/                  # User interfaces
│       └── streamlit_app.py # Web interface
├── data/                    # Data directory
│   ├── documents/           # Input documents
│   ├── faiss_index/         # Vector store
│   └── sample_docs/         # Sample documents
├── main.py                  # Entry point
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🔧 Technical Components

### Core Technologies (All Free!)
- **Sentence Transformers**: `all-MiniLM-L6-v2` for embeddings
- **Microsoft DialoGPT**: Small/medium models for text generation
- **FAISS**: CPU version for vector similarity search
- **LangChain**: RAG pipeline orchestration
- **Streamlit**: Web interface framework

### Data Science Techniques
- **Text Preprocessing**: Document parsing and cleaning
- **Text Chunking**: Recursive character splitting with overlap
- **Vector Embeddings**: Semantic representation of text
- **Similarity Search**: Cosine similarity in vector space
- **Retrieval-Augmented Generation**: Context-aware answer generation

## 🚀 Quick Start

### 1. Installation

```bash
# Navigate to project directory
cd sampleprojects

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the System

```bash
# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Run the system (defaults to demo mode)
python main.py
```

That's it! The system will:
- ✅ Create sample documents automatically
- ✅ Initialize the RAG system
- ✅ Process documents and create embeddings
- ✅ Ask sample questions and show answers
- ✅ Display enhanced logs with [filename:line_number]

## 📖 Adding Your Own Documents

Want to use your own PDF or documents instead of the sample files?

### Option 1: Add to Sample Documents Folder

```bash
# Copy your PDF to the documents folder
cp /path/to/your/document.pdf data/sample_docs/

# Run the system
python main.py
```

### Option 2: Use a Custom Folder

```python
from src.rag_qa import RAGDocumentQA

# Initialize system
rag = RAGDocumentQA()

# Ingest your documents from any folder
rag.ingest_documents("path/to/your/documents/")

# Ask questions about your documents
result = rag.answer_question("What is this document about?")
print(f"Answer: {result['answer']}")
```

**Supported formats:** PDF, DOCX, TXT, Markdown

## 🎛️ Advanced Options

### Debug Mode with Detailed Logs

```bash
python main.py --log-level DEBUG
```

### Save Logs to File

```bash
python main.py --log-level INFO --log-file rag_system.log
```

### Different Modes

```bash
python main.py demo          # Demo mode (default)
python main.py web           # Web interface (Streamlit)
python main.py interactive   # Interactive Q&A mode
```

## 🎯 Use Cases

### Healthcare
- **Medical Literature Review**: Query research papers and clinical guidelines
- **Patient Information**: Extract relevant information from medical records
- **Drug Information**: Search pharmaceutical documentation

### Finance
- **Regulatory Compliance**: Query financial regulations and policies
- **Market Research**: Analyze financial reports and market data
- **Risk Assessment**: Review risk management documents

### Education
- **Course Materials**: Search through textbooks and lecture notes
- **Research Assistance**: Query academic papers and publications
- **Study Guides**: Extract key information from educational content

### Enterprise
- **Knowledge Management**: Search internal documentation and policies
- **Technical Documentation**: Query API docs, manuals, and guides
- **Customer Support**: Find relevant information from support materials

## 🔬 Data Science Applications

This project demonstrates several key Data Science concepts:

### 1. Natural Language Processing
- **Text Preprocessing**: Cleaning and normalizing document content
- **Tokenization**: Breaking text into meaningful units
- **Embeddings**: Converting text to numerical representations

### 2. Information Retrieval
- **Vector Similarity**: Finding semantically similar content
- **Ranking**: Ordering results by relevance
- **Query Expansion**: Enhancing search queries

### 3. Machine Learning
- **Unsupervised Learning**: Clustering similar documents
- **Similarity Metrics**: Cosine similarity for vector comparison
- **Model Evaluation**: Measuring system performance

### 4. System Design
- **Scalability**: Efficient vector storage and retrieval
- **Modularity**: Separated concerns and reusable components
- **Configuration**: Flexible parameter management

## 🛠️ Development

### Adding New Document Types

```python
# In src/rag_qa/core/document_processor.py
def load_custom_format(self, file_path: str) -> str:
    """Load custom document format."""
    # Implement custom loading logic
    return extracted_text
```

### Custom Embedding Models

```python
# In src/rag_qa/core/rag_system.py
rag = RAGDocumentQA(
    embedding_model_name="your-custom-model",
    # other parameters...
)
```

### Extending Evaluation Metrics

```python
# In src/rag_qa/utils/evaluation.py
def custom_evaluation_metric(results):
    """Implement custom evaluation logic."""
    # Your evaluation code here
    return metric_value
```

## 🔍 Troubleshooting

### Module Not Found Error
Make sure you activated the virtual environment:
```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python main.py
```

### Out of Memory
The system will download AI models on first run. This is normal and only happens once. Models are cached for future use.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Include tests and documentation
5. Submit a pull request

## 📄 License

This project is open-source and available under the MIT License.

## 🙏 Acknowledgments

- **Hugging Face** for free transformer models
- **Facebook AI** for FAISS vector search
- **LangChain** for RAG framework
- **Streamlit** for web interface framework
- **Sentence Transformers** for embedding models

## 📚 References

1. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
2. [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
3. [LangChain Documentation](https://docs.langchain.com/)
4. [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss)

---

**🚀 Ready to build intelligent document QA systems with free resources!** 