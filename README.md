# 🔍 AI Code Analyzer

An intelligent code analysis tool that uses AI to help you understand, explore, and get insights from your codebase through natural language queries.

## 🌟 Features

- **Natural Language Queries**: Ask questions about your codebase in plain English
- **Multi-Repository Support**: Analyze multiple Git repositories simultaneously
- **Interactive Web Interface**: User-friendly Streamlit-based chat interface
- **Source Attribution**: Get references to specific files and code sections
- **Real-time Processing**: Add new repositories on-the-fly
- **AI-Powered Insights**

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Azure OpenAI API access
- Git (for repository cloning)

### Installation

1. **Clone the repository**
   ```bash
   https://github.com/Prathyusha-Project/assignment_AI_Codeagent.git
   cd codebase
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   # Azure OpenAI Configuration
   OPENAI_API_TYPE=azure
   OPENAI_API_VERSION=2024-06-01
   AZURE_OPENAI_ENDPOINT=your-azure-endpoint
   AZURE_OPENAI_API_KEY=your-api-key
   EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-large
   CHAT_DEPLOYMENT_NAME=gpt-4
   
   # Storage Configuration
   CHROMADB_PERSIST_DIRECTORY=./data/vector_db
   REPOSITORIES_PATH=./data/repositories
   
   # Processing Settings
   MAX_FILE_SIZE=50000
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Access the web interface**
   
   Open your browser and navigate to `http://localhost:8501`

## 📁 Project Structure

```
ai-code-analyzer/
├── core/                          # Core application modules
│   ├── doc_processor.py          # Document processing logic
│   ├── git_manager.py            # Git repository management
│   ├── rag_chain.py              # RAG (Retrieval-Augmented Generation) chain
│   ├── startup_processor.py      # Application initialization
│   └── vector_store.py           # Vector database operations
├── data/                         # Data storage
│   ├── repositories/             # Cloned repositories
│   └── vector_db/               # ChromaDB vector database
├── utils/                        # Utility modules
│   └── config.py                # Configuration management
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── .env                         # Environment variables (create this)
└── README.md                    # This file
```

## 🎯 Usage

### Initial Setup

1. **Start the application** - The app will initialize with default repositories
2. **Wait for processing** - Initial setup processes your codebase and creates embeddings
3. **Start asking questions** - Use the chat interface to query your code

### Adding Repositories

Use the sidebar to add additional repositories:

1. Enter the repository URL (e.g., `https://github.com/username/repo`)
2. Click "Add Repository"
3. Wait for processing to complete


## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AZURE_OPENAI_ENDPOINT` | Your Azure OpenAI endpoint URL | Required |
| `AZURE_OPENAI_API_KEY` | Your Azure OpenAI API key | Required |
| `EMBEDDING_DEPLOYMENT_NAME` | Embedding model deployment | `text-embedding-3-large` |
| `CHAT_DEPLOYMENT_NAME` | Chat model deployment | `gpt-4` |
| `CHROMADB_PERSIST_DIRECTORY` | Vector database storage path | `./data/vector_db` |
| `REPOSITORIES_PATH` | Repository storage path | `./data/repositories` |
| `MAX_FILE_SIZE` | Maximum file size to process (bytes) | `50000` |
| `CHUNK_SIZE` | Text chunk size for embeddings | `1000` |
| `CHUNK_OVERLAP` | Overlap between text chunks | `200` |

### Supported File Types

The analyzer processes common code and documentation files:
- Programming languages: `.py`, `.js`, `.java`, `.cpp`, `.c`, `.go`, `.rs`, etc.
- Documentation: `.md`, `.txt`, `.rst`
- Configuration: `.json`, `.yaml`, `.yml`, `.toml`
- Web: `.html`, `.css`

## 🔧 Core Components

### Document Processor (`core/doc_processor.py`)
Handles file reading, text extraction, and chunking for various file types.

### Git Manager (`core/git_manager.py`)
Manages repository cloning, updates, and file system operations.

### Vector Store (`core/vector_store.py`)
Manages ChromaDB operations, embeddings storage, and similarity search.

### RAG Chain (`core/rag_chain.py`)
Implements the Retrieval-Augmented Generation pipeline for answering queries.

### Startup Processor (`core/startup_processor.py`)
Handles application initialization, repository setup, and background processing.

## 🛠️ Development

### Adding New File Types

To support additional file types, modify `core/doc_processor.py`:

```python
def is_supported_file(self, file_path: str) -> bool:
    supported_extensions = {
        '.py', '.js', '.java', '.cpp', '.c', '.go', 
        '.rs', '.md', '.txt', '.json', '.yaml', '.yml'
        # Add your extensions here
    }
    return Path(file_path).suffix.lower() in supported_extensions
```

### Customizing Chunk Size

Adjust chunking parameters in your `.env` file:
- Increase `CHUNK_SIZE` for longer context
- Adjust `CHUNK_OVERLAP` to maintain context continuity

## 📊 Performance Tips

- **Repository Size**: Larger repositories take longer to process initially
- **File Filtering**: The system automatically filters out binary files and large files
- **Memory Usage**: ChromaDB stores embeddings in memory for fast retrieval
- **API Limits**: Be aware of Azure OpenAI rate limits for large repositories