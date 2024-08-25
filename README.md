# Quill - Chat with PDF

This is a Python-based web application that allows users to interact with PDF files through chat. The project leverages various AI and machine learning libraries to process and respond to user queries related to the content of the uploaded PDFs.

## Features

- **PDF Parsing:** Extract text from PDF files using `PyPDF2`.
- **AI-Powered Chat:** Utilize `google-generativeai` and `langchain` to generate conversational responses.
- **Streamlit Interface:** Provide a simple and interactive web UI using `streamlit`.
- **Efficient Search:** Integrate `faiss-cpu` for efficient text search and retrieval within PDF documents.
- **Environment Variables Management:** Securely manage API keys and environment variables using `python-dotenv`.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Yash-Tibadiya/quill.git
   cd quill

2. **Create a virtual environment and activate it:**
   ```bash
   python -m venv venv
   source venv/bin/activate

  - On Windows, use: venv\Scripts\activate

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt

4. **Set up environment variables:**
   - Create a .env file in the project root.
   - Add your Google Generative AI API key.

5. **Run the application:**
   ```bash
   streamlit run app.py


# Usage
- Upload a PDF file via the Streamlit interface.
- Ask questions related to the content of the uploaded PDF.
- The AI-powered chat will respond with relevant information based on the content of the PDF.
     

   
