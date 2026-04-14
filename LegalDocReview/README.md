# Legal Document Review Application

## Project Overview

The **Legal Document Review Application** is an intelligent, AI-powered tool designed to analyze, review, and extract key information from legal documents. Built with Streamlit, LangChain, and Azure OpenAI, this application provides structured legal document analysis with the capability to store and retrieve analysis results through a vector database.

### Key Features

- 📄 **Multi-Format Document Support**: Process PDF, DOCX, and TXT files
- 🤖 **AI-Powered Analysis**: Leverages Azure OpenAI's GPT-4 models for expert legal analysis
- 🔍 **Semantic Search**: Uses vector embeddings for document retrieval and similarity matching
- 💾 **Persistent Storage**: Stores analysis results in Chroma vector database for future reference
- 📥 **Download Capability**: Export analysis results as text files
- 🎯 **Query-Specific Analysis**: Analyze documents based on specific user queries
- 🏗️ **LangChain Integration**: Built with LangChain Expression Language (LCEL) for composable AI pipelines

---

## Author Information

**Author**: Shubham Guha  
**Training**: Edureka  
**AI Assistance**: GitHub Copilot  
**Last Updated**: April 2026

---

## Architecture & Technology Stack

### Frontend
- **Streamlit**: Interactive web-based user interface
- **Python 3.8+**: Core programming language

### Core Libraries
- **LangChain**: AI framework for composable LLM workflows
  - `langchain-core`: Core abstractions and interfaces
  - `langchain-community`: Community integrations
  - `langchain-openai`: Azure OpenAI integrations
- **OpenAI Python Client**: Azure OpenAI API integration
- **PyPDF**: PDF document processing
- **python-docx**: DOCX document processing
- **docx2txt**: DOC/DOCX text extraction

### Vector Database & Embeddings
- **Chroma**: Vector database for semantic search
  - Persistent storage: SQLite backend
  - In-memory and persistent modes
- **Azure OpenAI Embeddings**: text-embedding-3-large model for document vectorization

### Cloud Services
- **Azure OpenAI**: GPT-4 mini models for legal analysis
- **Azure OpenAI Embeddings**: For semantic document representations

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Azure OpenAI API keys
- `.env` file with proper configuration

### Step 1: Clone or Navigate to Project Directory
```bash
cd LegalDocReview
```

### Step 2: Create Virtual Environment
```bash
python -m venv v_env
```

### Step 3: Activate Virtual Environment

#### On Windows (PowerShell):
```powershell
.\v_env\Scripts\Activate.ps1
```

#### On Windows (Command Prompt):
```cmd
.\v_env\Scripts\activate.bat
```

#### On macOS/Linux:
```bash
source v_env/bin/activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Configure Environment Variables

Create a `.env` file in the project root directory:

```env
AZURE_OPENAI_ENDPOINT=https://<your-resource-name>.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
```

**Security Note**: Never commit the `.env` file to version control. Add it to `.gitignore`.

### Step 6: Run the Application
```bash
streamlit run legal_doc_app.py
```

The application will open at `http://localhost:8501`

---

## Project Structure

```
LegalDocReview/
├── legal_doc_app.py           # Main Streamlit application
├── requirements.txt            # Python package dependencies
├── README.md                   # This file
├── .env                        # Environment variables (not in repo)
├── .gitignore                  # Git ignore rules
├── chroma_store/               # Vector database storage
│   ├── chroma.sqlite3          # Chroma metadata database
│   └── [UUID directories]/     # Embedded documents storage
├── v_env/                      # Virtual environment
└── sample_documents/           # Sample legal documents for testing
```

---

## Usage Guide

### Basic Workflow

1. **Start the Application**
   ```bash
   streamlit run legal_doc_app.py
   ```

2. **Upload a Legal Document**
   - Click "Upload Document" in the right column
   - Select a PDF, DOCX, or TXT file containing a legal document
   - File will be temporarily processed and automatically cleaned up

3. **Enter Your Query**
   - In the left column, enter specific questions or areas of interest
   - Example queries:
     - "What are the termination clauses in this agreement?"
     - "Identify all liability limitations in this contract"
     - "What are the key rights and obligations of each party?"

4. **Generate Analysis**
   - Click the "Analyze" button
   - The AI will process your document and query
   - Analysis results will appear with key clauses and summary

5. **View Document Text**
   - Expand the "View Document Text" section to see extracted content
   - Useful for verification and understanding the source material

6. **Download Results**
   - Click "Download Analysis" to save the analysis as a text file
   - Results are automatically stored in the vector database

### Advanced Features

#### Vector Database Persistence
- All analyses are automatically stored in the Chroma vector database
- Enables semantic search for historical analyses
- Data persists between application sessions

#### Semantic Search Capability
- Future queries can leverage historical analyses
- Find similar documents and analyses through semantic similarity
- Build a knowledge base of reviewed documents

---

## Dependencies

### Core Dependencies
```
streamlit              # Web UI framework
langchain              # AI framework
langchain-core         # LangChain core components
langchain-community    # Community integrations
python-dotenv          # Environment variable management
docx2txt              # DOCX file processing
pypdf                 # PDF file processing
openai                # Azure OpenAI API client
langchain-openai      # LangChain OpenAI integration
langchain-text-splitters  # Text chunking utilities
langchain-classic     # LangChain classic chains
chromadb              # Vector database
```

See `requirements.txt` for the complete list with versions.

---

## Configuration

### Azure OpenAI Configuration

The application requires Azure OpenAI credentials. Configure these in your `.env` file:

```env
AZURE_OPENAI_ENDPOINT=https://<resource-name>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-32-character-api-key>
```

### LLM Model Parameters

The application uses the following model configurations:

- **Model**: GPT-4.1-mini (configurable via `azure_deployment`)
- **Temperature**: 0.2 (low for consistent legal analysis)
- **API Version**: 2024-02-15-preview
- **Embedding Model**: text-embedding-3-large

### Text Splitting Parameters

- **Chunk Size**: 500 characters per chunk
- **Chunk Overlap**: 50 characters (for context preservation)

Adjust these in the `split_text()` function for different document types.

---

## Application Flow

```
┌─────────────────────────────────┐
│   User Uploads Document         │
│   + Enters Query                │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Extract Text from Document    │
│   (PDF/DOCX/TXT)                │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Create LangChain Pipeline     │
│   - Map inputs                  │
│   - Apply prompt template       │
│   - Send to Azure OpenAI        │
│   - Parse output                │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Generate AI Analysis          │
│   - Extract key clauses         │
│   - Provide concise summary     │
│   - Answer specific query       │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Store in Vector Database      │
│   - Chunk analysis              │
│   - Create embeddings           │
│   - Persist to Chroma           │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Display Results to User       │
│   - Show analysis               │
│   - Provide download option     │
└─────────────────────────────────┘
```

---

## Code Structure

### Main Components

#### 1. **Initialization & Configuration**
```python
# Environment setup
# Azure OpenAI client initialization
# Embedding model setup
# Vector database initialization
```

#### 2. **Document Processing Functions**
- `extract_text_from_legal_doc(file)`: Convert uploaded files to text
- `split_text(text)`: Chunk text for efficient processing
- `store_doc_analysis(resume_text, analysis, doc_id)`: Persist analysis to database

#### 3. **Main Application**
- `main()`: Orchestrates the Streamlit UI and analysis workflow
- Two-column layout for optimal UX
- Real-time analysis with loading indicators

---

## Performance Considerations

### Document Size
- Optimal **document size**: 5KB - 100KB
- Maximum recommended: 500KB
- Larger documents may require longer processing time

### Chunking Strategy
- Current configuration handles documents up to ~10,000 characters effectively
- Adjust `chunk_size` and `chunk_overlap` for different document types

### API Rate Limits
- Respect Azure OpenAI API rate limits
- Monitor usage in Azure portal
- Consider batch processing for high-volume scenarios

---

## Troubleshooting

### Common Issues

#### 1. "Please set your AZURE_API_KEY in a .env file"
- **Solution**: Ensure `.env` file exists in the project root
- Verify `AZURE_OPENAI_API_KEY` is correctly set
- No spaces around the `=` sign

#### 2. "Unsupported file format"
- **Solution**: Only PDF, DOCX, and TXT files are supported
- Convert other formats before uploading

#### 3. Vector Database Connection Error
- **Solution**: Check write permissions for `chroma_store` directory
- Delete `chroma_store` directory to reset the database
- Restart the application

#### 4. Slow Processing
- **Solution**: Check internet connection to Azure OpenAI
- Monitor OpenAI API response times
- Process smaller documents first

---

## Best Practices

### Document Preparation
1. Ensure document is readable and not image-based
2. Remove excessive branding/headers if possible
3. Use clear, structured formatting

### Queries
1. **Be Specific**: Ask targeted questions about specific clauses
2. **Use Context**: Provide relevant background information
3. **Structure**: Ask one question at a time for best results

### Data Management
1. **Backup**: Regularly backup the `chroma_store` directory
2. **Privacy**: Don't upload sensitive personal information unnecessarily
3. **Cleanup**: Periodically review and delete stale analyses

---

## API Reference

### Function Documentation

#### `extract_text_from_legal_doc(file) -> str`
Extracts text from uploaded documents (PDF, DOCX, TXT)

#### `split_text(text) -> List[Document]`
Splits text into chunks for vectorization

#### `store_doc_analysis(resume_text, analysis, doc_id) -> None`
Stores analysis results in vector database

#### `main() -> None`
Main Streamlit application orchestrator

---

## Future Enhancements

- 🔐 **Authentication**: Add user authentication for multi-user support
- 📊 **Analytics Dashboard**: Track document analysis metrics
- 🌐 **Multi-Language Support**: Support documents in multiple languages
- ⚡ **Batch Processing**: Process multiple documents simultaneously
- 🔄 **RAG System**: Implement Retrieval-Augmented Generation for context-aware analysis
- 📱 **Mobile UI**: Responsive design for mobile devices
- 🔗 **Integration**: REST API for third-party integrations

---

## Security Considerations

1. **API Keys**: Store Azure OpenAI keys securely in `.env` files
2. **Data Privacy**: Be cautious with sensitive legal documents
3. **Temporary Files**: All temporary files are automatically cleaned up
4. **Vector Database**: Protect `chroma_store` directory from unauthorized access
5. **Rate Limiting**: Implement rate limiting for production deployments

---

## Legal Disclaimer

This application is provided for informational and analytical purposes only. While it uses advanced AI models, it should not be considered as a substitute for professional legal advice. Always consult with qualified legal professionals for important legal matters.

---

## License

This project is provided under an educational license for learning and research purposes.

---

## Support & Contact

For issues, questions, or suggestions:
1. Review the troubleshooting section above
2. Check application logs for error messages
3. Verify Azure OpenAI configuration
4. Contact: Shubham Guha

---

## Acknowledgments

- **Edureka**: Training and foundational knowledge
- **GitHub Copilot**: AI assistance in code development and documentation
- **LangChain**: Excellent framework for LLM applications
- **Azure OpenAI**: Powerful language models and embeddings
- **Streamlit**: Rapid UI development framework

---

## version History

- **v1.0** (April 2026): Initial release with core functionality
  - Document upload and text extraction
  - AI-powered legal analysis
  - Vector database storage
  - Download capability

---

**Last Updated**: April 14, 2026  
**Author**: Shubham Guha  
**Status**: Active Development
