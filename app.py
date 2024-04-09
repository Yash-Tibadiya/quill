# run command : 
# .\venv\Scripts\activate
# streamlit run app.py 


import streamlit as st
# Streamlit is a free, open-source app framework that allows users to create interactive web apps from Python scripts.
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
# Google generative AI, or generative artificial intelligence, is a machine learning model that can create new content like images, music, text, videos, and audio
from langchain.vectorstores import FAISS
# Facebook AI Similarity Search (FAISS) is a library that allows developers to quickly search for similar multimedia document embeddings.
# Vector embeddings are a machine learning process that transforms data like words, sentences, and images into numbers that represent their meaning and relationships. 
from langchain_google_genai import ChatGoogleGenerativeAI
# Access Google AI’s gemini and gemini-vision models, as well as other generative models through ChatGoogleGenerativeAI class in the langchain-google-genai integration package.
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# LangChain is an open-source framework that helps developers create applications using large language models (LLMs). LLMs are deep-learning models that are pre-trained on large amounts of data and can respond to user queries. LangChain provides tools and abstractions to improve the customization, accuracy, and relevancy of the information the models generate.
# Faiss-cpu is a library for similarity search and clustering of dense vectors



# read environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# convert all thing of pdf in text and read all pdf pages
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text


# divide all text in to chanks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


# convert chanks into vector and store on local
# store in one local folder in format of pkl(pikals) so not readable
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# create prompt template which will conversion with google gemini pro 
def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


# load user question and pdf vectors from local (text box)
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])




def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF 💬")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()