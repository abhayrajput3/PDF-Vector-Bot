"""
Example usage of the Gemini PDF Analyzer

This script demonstrates how to use the GeminiPDFProcessor class programmatically.
"""

import os
from gemini_pdf_analyzer import GeminiPDFProcessor

def example_usage():
    """Example of how to use the Gemini PDF analyzer programmatically"""

    # Option 2: Pass directly to constructor
    api_key = "AIzaSyAN7jezELrGX1taMIsG7ipmYWplIy9VtJ0"  # Replace with your actual API key
    # api_key = "AIzaSyBArD-OeKCGN3NqVWeuE8mx3oKJzv_KKyw"  # Replace with your actual API key

    try:
        # Initialize the processor with Gemini
        processor = GeminiPDFProcessor(google_api_key=api_key)

        # Process a PDF file
        pdf_path = r"C:\Users\abhay\Downloads\23c4b858-2b24-4d5a-a7b5-da6d755b8a48.pdf"  # Replace with your PDF path

        if os.path.exists(pdf_path):
            # Load and process the PDF
            success = processor.process_pdf(pdf_path)

            if success:
                print("PDF processed successfully with Gemini!")

                # Generate summary using Gemini
                print("\nGenerating summary with Gemini...")
                summary = processor.generate_summary(processor.documents)
                print("\nGEMINI SUMMARY:")
                print(summary)

                # Ask questions using Gemini
                questions = [
                    "What is the main topic of this document?",
                    "What are the key findings?",
                    "Who are the authors?",
                    "What conclusions are drawn?"
                ]

                for question in questions:
                    print(f"\nQUESTION: {question}")
                    result = processor.ask_question(question)
                    print(f"GEMINI ANSWER: {result['answer']}")

                # Direct Gemini chat example
                print("\nDirect Gemini API call example:")
                direct_response = processor.direct_gemini_call("Explain the benefits of using Google Gemini API")
                print("GEMINI RESPONSE:", direct_response)

            else:
                print("Failed to process PDF with Gemini")

        else:
            print(f"PDF file not found: {pdf_path}")

    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure your Google API key is valid and has access to Gemini models")

if __name__ == "__main__":
    example_usage()
