# PDF Analyzer with Google Gemini API

A comprehensive Python application that extracts text from PDF files, generates summaries, and answers questions based on the content using Google's Gemini AI through the LangChain framework.

## 🚀 Features

🤖 **Google Gemini Integration**: Uses Google's latest Gemini 2.0 Flash model for text processing
🔍 **PDF Text Extraction**: Extract and process text from PDF documents using PyPDF
📝 **Intelligent Summarization**: Generate summaries using different strategies with Gemini AI
❓ **Question Answering**: Ask questions about the PDF content using RAG with Gemini
🌐 **Web Interface**: User-friendly Streamlit web interface
💻 **CLI Interface**: Command-line interface for programmatic usage
⚡ **Vector Search**: Efficient similarity search using FAISS with Gemini embeddings
💬 **Direct Chat**: Direct API calls to Gemini for additional insights

## 🏗️ Architecture

The application uses the following components:

- **Document Loaders**: PyPDFLoader for PDF text extraction
- **Text Splitters**: RecursiveCharacterTextSplitter for chunking documents
- **Embeddings**: Google Generative AI embeddings (text-embedding-004)
- **Vector Store**: FAISS for efficient similarity search
- **LLM Integration**: Google Gemini 2.0 Flash for summarization and Q&A
- **Chains**: Summarization and RetrievalQA chains with Gemini

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key (free from Google AI Studio)

### Setup

1. **Install dependencies**:
   ```bash
   pip install -r gemini_requirements.txt
   ```

2. **Get your Google API key**:
   - Visit [Google AI Studio](https://ai.google.dev)
   - Sign in with your Google account
   - Click "Create API key"
   - Copy your API key

3. **Set up environment variables**:
   ```bash
   # Copy the template
   cp gemini.env.template .env

   # Edit .env and add your Google API key
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

## 🎯 Usage

### Web Interface (Recommended)

Run the Streamlit web application:

```bash
streamlit run gemini_pdf_analyzer.py --streamlit
```

Features include:
- Upload PDF files through browser
- View document information
- Generate summaries with different Gemini strategies
- Ask questions about the content using Gemini
- Direct chat with Gemini for additional insights

### Command Line Interface

Run the CLI version:

```bash
python gemini_pdf_analyzer.py
```

Options:
1. Generate summary with Gemini (stuff, map-reduce, or refine)
2. Ask questions about the PDF content
3. Direct chat with Gemini
4. Exit

### Programmatic Usage

```python
from gemini_pdf_analyzer import GeminiPDFProcessor

# Initialize with Gemini
processor = GeminiPDFProcessor("your-google-api-key")

# Process PDF
processor.process_pdf("document.pdf")

# Generate summary with Gemini
summary = processor.generate_summary(processor.documents)
print(summary)

# Ask questions using Gemini
result = processor.ask_question("What is the main topic?")
print(result["answer"])

# Direct Gemini API call
response = processor.direct_gemini_call("Explain this in simple terms")
print(response)
```

## ⚙️ Configuration

### Gemini Models Used

- **Text Generation**: `gemini-2.0-flash` (latest and most capable)
- **Embeddings**: `models/text-embedding-004` (768 dimensions)

### Summarization Types

- **stuff**: Simple concatenation (fast, good for small documents)
- **map_reduce**: Parallel processing (best for large documents)  
- **refine**: Iterative refinement (highest quality)

### Customizable Parameters

```python
# Text splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Adjust chunk size
    chunk_overlap=200,    # Adjust overlap
)

# Gemini model settings
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,      # Control randomness
    max_retries=2,        # Retry failed requests
)

# Vector search
retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}  # Number of similar chunks
)
```

## 🔧 API Key Setup

### Free Tier Limits

Google AI Studio provides generous free tier limits:
- 15 requests per minute
- 1,500 requests per day
- 32,000 tokens per minute

### Getting Started

1. Go to [Google AI Studio](https://ai.google.dev)
2. Sign in with Google account
3. Accept terms of service
4. Generate API key
5. Add to `.env` file or environment variables

### Environment Variables

**Option 1: .env file**
```bash
GOOGLE_API_KEY=your_api_key_here
```

**Option 2: System environment**
```bash
export GOOGLE_API_KEY=your_api_key_here
```

**Option 3: Direct in code**
```python
processor = GeminiPDFProcessor(google_api_key="your_api_key_here")
```

## 🚀 Advantages of Gemini vs OpenAI

### Cost Benefits
- **Free Tier**: Generous free usage limits
- **Competitive Pricing**: Lower cost per token for paid usage
- **No Monthly Minimums**: Pay only for what you use

### Performance Benefits
- **Multimodal Native**: Built-in support for text, images, audio, video
- **Large Context Window**: Handle longer documents efficiently  
- **Fast Inference**: Optimized for speed and efficiency
- **Latest Technology**: Access to Google's cutting-edge AI research

### Integration Benefits
- **Google Ecosystem**: Works seamlessly with Google services
- **Regular Updates**: Frequent model improvements and new features
- **Reliable Infrastructure**: Built on Google's robust infrastructure

## 🔍 Troubleshooting

### Common Issues

**"API key not valid"**
- Ensure your Google API key is correct
- Check that the API key has access to Gemini models
- Verify the key is set in environment variables

**"Rate limit exceeded"**
- Wait a moment before retrying
- Consider upgrading to paid tier for higher limits
- Implement request throttling in your code

**"Model not found"**
- Ensure you're using correct model names
- Check if the model is available in your region
- Try using `gemini-pro` as a fallback

### Performance Tips

- **Large PDFs**: Use "map_reduce" summarization
- **Better Embeddings**: Use the latest embedding models
- **Faster Processing**: Reduce chunk_size for quicker responses
- **Cost Optimization**: Use appropriate models for your use case

## 📁 File Structure

```
gemini-pdf-analyzer/
├── gemini_pdf_analyzer.py      # Main application with Gemini
├── gemini_requirements.txt     # Dependencies for Gemini
├── gemini.env.template         # Environment template
├── gemini_example_usage.py     # Example usage
└── README_GEMINI.md           # This documentation
```

## 🤝 Comparison with OpenAI Version

| Feature | OpenAI Version | Gemini Version |
|---------|----------------|----------------|
| API Cost | Paid only | Free tier + paid |
| Model Performance | GPT-3.5/4 | Gemini 2.0 Flash |
| Context Window | 4K-128K tokens | Up to 1M tokens |
| Multimodal | Limited | Native support |
| Rate Limits | Token-based | Request-based |
| Setup Complexity | Medium | Simple |

## 📈 Next Steps

Potential enhancements:
- Add support for multimodal inputs (images in PDFs)
- Implement caching for repeated queries
- Add batch processing for multiple PDFs
- Create vector database persistence
- Add support for other document formats

## 📄 License

This project is provided as-is for educational purposes.

## 🆘 Support

- **Google AI Studio**: [https://ai.google.dev](https://ai.google.dev)
- **Gemini API Docs**: [https://ai.google.dev/docs](https://ai.google.dev/docs)
- **LangChain Google Integration**: [https://python.langchain.com/docs/integrations/providers/google/](https://python.langchain.com/docs/integrations/providers/google/)

---

**Ready to analyze PDFs with Google Gemini? Get started now! 🚀**
