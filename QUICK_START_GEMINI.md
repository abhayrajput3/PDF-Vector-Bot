# 🚀 Quick Start - PDF Analyzer with Google Gemini

## Get Started in 3 Minutes!

### Step 1: Install Dependencies
```bash
pip install -r gemini_requirements.txt
```

### Step 2: Get Your FREE Google Gemini API Key
1. 🌐 Visit [Google AI Studio](https://ai.google.dev)
2. 🔐 Sign in with your Google account
3. ✨ Click "Create API key" (it's FREE!)
4. 📋 Copy your API key

### Step 3: Set Up Environment
```bash
# Copy the template
cp gemini.env.template .env

# Add your API key to .env
echo "GOOGLE_API_KEY=your_actual_api_key_here" > .env
```

### Step 4: Run the Application

**🌐 Web Interface (Recommended)**
```bash
streamlit run gemini_pdf_analyzer.py --streamlit
```
Then open http://localhost:8501

**💻 Command Line**
```bash
python gemini_pdf_analyzer.py
```

### Step 5: Start Analyzing!

#### Web Interface:
1. 🔑 Enter your Google API key in the sidebar
2. 📎 Upload a PDF file
3. 🎯 Choose from 4 tabs:
   - **📄 Document Info**: Basic information
   - **📝 Summary**: AI-generated summaries  
   - **❓ Q&A**: Ask questions about content
   - **🧠 Direct Chat**: Chat directly with Gemini

#### Command Line:
1. 📁 Enter PDF file path
2. 🎯 Choose options:
   - Generate summary
   - Ask questions
   - Direct Gemini chat
   - Exit

## 💡 Example Usage

```python
from gemini_pdf_analyzer import GeminiPDFProcessor

# Initialize with Gemini (FREE!)
processor = GeminiPDFProcessor("your-google-api-key")

# Process PDF
processor.process_pdf("document.pdf")

# Get summary with Gemini
summary = processor.generate_summary(processor.documents)
print(summary)

# Ask questions using Gemini's intelligence
result = processor.ask_question("What are the main conclusions?")
print(result["answer"])

# Chat directly with Gemini
response = processor.direct_gemini_call("Explain this document's significance")
print(response)
```

## 🎁 Why Choose Gemini?

- ✅ **FREE Tier**: Generous limits for getting started
- ✅ **Latest AI**: Google's most advanced Gemini 2.0 Flash
- ✅ **Large Context**: Handle longer documents (1M tokens!)
- ✅ **Fast & Reliable**: Built on Google's infrastructure
- ✅ **Multimodal Ready**: Future-proof for images, audio, video

## 🆘 Need Help?

**API Key Issues?**
- Make sure you copied the complete key
- Check it's set in .env file correctly
- Verify you have internet connection

**PDF Not Loading?**
- Ensure PDF contains text (not just images)
- Check file path is correct
- Try with a smaller PDF first

**Rate Limits?**
- Free tier: 15 requests/minute, 1,500/day
- Wait a moment between requests
- Consider upgrading if needed

## 🎯 Pro Tips

- 📄 **Large PDFs**: Use "map_reduce" summarization
- 💰 **Cost Control**: Free tier is generous for most users
- 🎯 **Better Questions**: Be specific for better answers
- ⚡ **Performance**: Smaller chunks = faster processing

## 🚀 Ready to Go!

You're now ready to analyze PDFs with Google Gemini! 

Start with a simple PDF and explore all the features. Gemini's advanced AI will provide insightful summaries and accurate answers to your questions.

**Happy analyzing! 🎉**
