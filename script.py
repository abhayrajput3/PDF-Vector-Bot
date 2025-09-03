# Create a modified version of the PDF analyzer using Gemini API instead of OpenAI

gemini_pdf_analyzer = '''
"""
PDF Text Extraction, Summarization, and Question-Answering Application
Using LangChain Framework with Google Gemini API

This application provides the following features:
1. Extract text from PDF files using PyPDF
2. Generate summaries of PDF content using Google Gemini
3. Answer questions based on PDF content using Google Gemini
4. Interactive command-line and Streamlit web interface

Requirements:
pip install langchain langchain-google-genai langchain-community pypdf faiss-cpu python-dotenv streamlit

Author: AI Assistant
Date: August 2025
"""

import os
import sys
from typing import List, Optional
from dotenv import load_dotenv
import streamlit as st
from io import StringIO
import tempfile
import requests
import json

# LangChain imports for Gemini
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Load environment variables
load_dotenv()

class GeminiPDFProcessor:
    """Main class for PDF processing, summarization, and Q&A using Google Gemini"""
    
    def __init__(self, google_api_key: str = None):
        """Initialize the PDF processor with Google Gemini API key"""
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        
        if not self.google_api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass it directly.")
        
        # Initialize LangChain components with Gemini
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=self.google_api_key,
            model="gemini-2.0-flash",  # Use the latest Gemini model
            temperature=0.3,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=self.google_api_key,
            model="models/text-embedding-004"  # Gemini embedding model
        )
        
        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        self.vector_store = None
        self.documents = None
        
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """Load PDF and extract text using PyPDFLoader"""
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            self.documents = documents
            
            print(f"Successfully loaded PDF: {pdf_path}")
            print(f"Number of pages: {len(documents)}")
            
            return documents
        
        except Exception as e:
            print(f"Error loading PDF: {str(e)}")
            return []
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks"""
        try:
            chunks = self.text_splitter.split_documents(documents)
            print(f"Split into {len(chunks)} chunks")
            return chunks
        
        except Exception as e:
            print(f"Error splitting documents: {str(e)}")
            return []
    
    def create_vector_store(self, chunks: List[Document]) -> FAISS:
        """Create vector store for similarity search using Gemini embeddings"""
        try:
            vector_store = FAISS.from_documents(chunks, self.embeddings)
            self.vector_store = vector_store
            print("Vector store created successfully with Gemini embeddings")
            return vector_store
        
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            return None
    
    def generate_summary(self, documents: List[Document], summary_type: str = "stuff") -> str:
        """Generate summary of the PDF content using Gemini"""
        try:
            # Custom prompt for summarization
            prompt_template = """
            Write a comprehensive summary of the following text.
            Focus on the main points, key findings, and important details.
            Provide a well-structured summary that captures the essence of the document.
            
            TEXT: {text}
            
            SUMMARY:
            """
            
            prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
            
            # Load summarization chain with Gemini
            chain = load_summarize_chain(
                self.llm,
                chain_type=summary_type,
                prompt=prompt,
                verbose=True
            )
            
            # Generate summary
            summary = chain.run(documents)
            return summary
        
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return "Error generating summary with Gemini."
    
    def setup_qa_chain(self) -> Optional[RetrievalQA]:
        """Setup question-answering chain with Gemini"""
        if not self.vector_store:
            print("Vector store not initialized. Please load and process a PDF first.")
            return None
        
        try:
            # Create retriever
            retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 4}  # Return top 4 most relevant chunks
            )
            
            # Custom prompt for Q&A with Gemini
            qa_prompt = PromptTemplate(
                template="""
                Use the following pieces of context to answer the question at the end.
                If you don't know the answer based on the context provided, just say that you don't know.
                Don't try to make up an answer.
                
                Context: {context}
                
                Question: {question}
                
                Answer: """,
                input_variables=["context", "question"]
            )
            
            # Create RetrievalQA chain with Gemini
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": qa_prompt}
            )
            
            return qa_chain
        
        except Exception as e:
            print(f"Error setting up Q&A chain: {str(e)}")
            return None
    
    def ask_question(self, question: str) -> dict:
        """Ask a question about the PDF content using Gemini"""
        qa_chain = self.setup_qa_chain()
        
        if not qa_chain:
            return {"answer": "Q&A system not available.", "sources": []}
        
        try:
            result = qa_chain.invoke({"query": question})
            
            return {
                "answer": result["result"],
                "sources": [doc.page_content[:200] + "..." for doc in result["source_documents"]]
            }
        
        except Exception as e:
            print(f"Error answering question: {str(e)}")
            return {"answer": "Error processing question with Gemini.", "sources": []}
    
    def direct_gemini_call(self, prompt: str) -> str:
        """Direct API call to Gemini for simple text generation"""
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
            
            headers = {
                'Content-Type': 'application/json',
                'X-goog-api-key': self.google_api_key
            }
            
            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ]
            }
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                return f"Error: {response.status_code} - {response.text}"
        
        except Exception as e:
            return f"Error making direct Gemini API call: {str(e)}"
    
    def process_pdf(self, pdf_path: str) -> bool:
        """Complete PDF processing pipeline with Gemini"""
        print("Starting PDF processing with Google Gemini...")
        
        # Load PDF
        documents = self.load_pdf(pdf_path)
        if not documents:
            return False
        
        # Split into chunks
        chunks = self.split_documents(documents)
        if not chunks:
            return False
        
        # Create vector store with Gemini embeddings
        vector_store = self.create_vector_store(chunks)
        if not vector_store:
            return False
        
        print("PDF processing completed successfully with Gemini!")
        return True


# Streamlit Web Interface
def main_streamlit():
    """Streamlit web interface for the Gemini PDF processor"""
    
    st.set_page_config(
        page_title="PDF Analyzer with Google Gemini",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 PDF Analyzer with Google Gemini")
    st.markdown("Extract text, generate summaries, and ask questions about your PDF documents using Google's Gemini AI")
    
    # Sidebar for API key input
    with st.sidebar:
        st.header("Configuration")
        google_api_key = st.text_input("Google API Key", type="password", help="Enter your Google Gemini API key from AI Studio")
        
        st.markdown("---")
        st.markdown("### 🔗 Get Your API Key")
        st.markdown("1. Visit [Google AI Studio](https://ai.google.dev)")
        st.markdown("2. Sign in with your Google account")
        st.markdown("3. Create a new API key")
        st.markdown("4. Copy and paste it above")
        
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
    
    # Main interface
    if not google_api_key:
        st.warning("Please enter your Google API key in the sidebar to continue.")
        st.info("Get your free API key from Google AI Studio: https://ai.google.dev")
        st.stop()
    
    try:
        # Initialize PDF processor with Gemini
        processor = GeminiPDFProcessor(google_api_key)
        
        # File upload
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                pdf_path = tmp_file.name
            
            # Process PDF
            with st.spinner("Processing PDF with Google Gemini..."):
                success = processor.process_pdf(pdf_path)
            
            if success:
                st.success("PDF processed successfully with Gemini!")
                
                # Create tabs for different functionalities
                tab1, tab2, tab3, tab4 = st.tabs(["📄 Document Info", "📝 Summary", "❓ Q&A", "🧠 Direct Chat"])
                
                with tab1:
                    st.header("Document Information")
                    if processor.documents:
                        st.write(f"**Number of pages:** {len(processor.documents)}")
                        st.write(f"**AI Model:** Google Gemini 2.0 Flash")
                        st.write(f"**Embedding Model:** Text Embedding 004")
                        
                        # Show first page preview
                        if len(processor.documents) > 0:
                            st.subheader("First Page Preview")
                            st.text_area(
                                "Content", 
                                processor.documents[0].page_content[:1000] + "..." if len(processor.documents[0].page_content) > 1000 else processor.documents[0].page_content,
                                height=300
                            )
                
                with tab2:
                    st.header("Document Summary with Gemini")
                    
                    summary_type = st.selectbox(
                        "Choose summary type",
                        ["stuff", "map_reduce", "refine"],
                        help="Different summarization strategies"
                    )
                    
                    if st.button("Generate Summary", type="primary"):
                        with st.spinner("Generating summary with Gemini..."):
                            summary = processor.generate_summary(processor.documents, summary_type)
                        
                        st.subheader("Summary")
                        st.write(summary)
                
                with tab3:
                    st.header("Ask Questions")
                    
                    # Question input
                    question = st.text_input("Enter your question about the PDF:")
                    
                    if st.button("Ask Gemini", type="primary") and question:
                        with st.spinner("Finding answer with Gemini..."):
                            result = processor.ask_question(question)
                        
                        st.subheader("Answer")
                        st.write(result["answer"])
                        
                        if result["sources"]:
                            with st.expander("Source References"):
                                for i, source in enumerate(result["sources"]):
                                    st.write(f"**Source {i+1}:** {source}")
                
                with tab4:
                    st.header("Direct Chat with Gemini")
                    st.markdown("Ask general questions or get additional insights using Gemini directly")
                    
                    direct_question = st.text_input("Ask Gemini anything:")
                    
                    if st.button("Chat with Gemini", type="primary") and direct_question:
                        with st.spinner("Asking Gemini..."):
                            response = processor.direct_gemini_call(direct_question)
                        
                        st.subheader("Gemini Response")
                        st.write(response)
            
            else:
                st.error("Failed to process PDF. Please check the file and try again.")
            
            # Clean up temporary file
            try:
                os.unlink(pdf_path)
            except:
                pass
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Make sure your Google API key is valid and has access to Gemini models.")


# Command Line Interface
def main_cli():
    """Command line interface for the Gemini PDF processor"""
    
    print("=== PDF Analyzer with Google Gemini ===")
    print("Extract text, generate summaries, and ask questions about PDF documents")
    print()
    
    # Get Google API key
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        google_api_key = input("Enter your Google Gemini API key: ").strip()
        if not google_api_key:
            print("Google API key is required. Exiting.")
            return
    
    try:
        # Initialize processor
        processor = GeminiPDFProcessor(google_api_key)
        
        # Get PDF path
        pdf_path = input("Enter the path to your PDF file: ").strip()
        
        if not os.path.exists(pdf_path):
            print("File not found. Please check the path and try again.")
            return
        
        # Process PDF
        print("\\nProcessing PDF with Google Gemini...")
        success = processor.process_pdf(pdf_path)
        
        if not success:
            print("Failed to process PDF. Exiting.")
            return
        
        while True:
            print("\\n=== Options ===")
            print("1. Generate Summary")
            print("2. Ask Question")
            print("3. Direct Gemini Chat")
            print("4. Exit")
            
            choice = input("\\nSelect an option (1-4): ").strip()
            
            if choice == "1":
                print("\\nSummary types:")
                print("1. Stuff (simple, for small documents)")
                print("2. Map-reduce (for large documents)")
                print("3. Refine (iterative refinement)")
                
                summary_choice = input("Select summary type (1-3): ").strip()
                summary_types = {"1": "stuff", "2": "map_reduce", "3": "refine"}
                summary_type = summary_types.get(summary_choice, "stuff")
                
                print(f"\\nGenerating {summary_type} summary with Gemini...")
                summary = processor.generate_summary(processor.documents, summary_type)
                
                print("\\n=== GEMINI SUMMARY ===")
                print(summary)
                print("=" * 50)
            
            elif choice == "2":
                question = input("\\nEnter your question: ").strip()
                if question:
                    print("\\nSearching for answer with Gemini...")
                    result = processor.ask_question(question)
                    
                    print("\\n=== GEMINI ANSWER ===")
                    print(result["answer"])
                    
                    if result["sources"]:
                        print("\\n=== SOURCES ===")
                        for i, source in enumerate(result["sources"]):
                            print(f"Source {i+1}: {source}")
                    
                    print("=" * 50)
            
            elif choice == "3":
                question = input("\\nAsk Gemini anything: ").strip()
                if question:
                    print("\\nAsking Gemini...")
                    response = processor.direct_gemini_call(question)
                    
                    print("\\n=== GEMINI RESPONSE ===")
                    print(response)
                    print("=" * 50)
            
            elif choice == "4":
                print("Thank you for using PDF Analyzer with Gemini!")
                break
            
            else:
                print("Invalid choice. Please try again.")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check your Google API key and internet connection.")


if __name__ == "__main__":
    # Check if running with Streamlit
    if len(sys.argv) > 1 and sys.argv[1] == "--streamlit":
        main_streamlit()
    else:
        # Run CLI version by default
        print("To run the web interface, use: streamlit run gemini_pdf_analyzer.py --streamlit")
        print("Running CLI version...")
        main_cli()
'''

# Save the Gemini version to a file
with open("gemini_pdf_analyzer.py", "w", encoding="utf-8") as f:
    f.write(gemini_pdf_analyzer)

print("✅ Gemini PDF Analyzer application created successfully!")
print("📁 File saved as: gemini_pdf_analyzer.py")