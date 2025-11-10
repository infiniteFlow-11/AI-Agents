from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import faiss, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import tempfile

load_dotenv()

gemeni_api_key = os.getenv("GEMINI_API_KEY")

# def get_pdf_text(pdf_files):
#     raw_text=""
#     for pdf in pdf_files:         
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             raw_text += page.extract_text()
#     return raw_text


# def get_pdf_text(pdf_docs):
#     raw_text = ""
#     for pdf in pdf_docs:
#         # Save the uploaded file to a temporary file
#         with tempfile.NamedTemporaryFile(delete=False) as f:
#             f.write(pdf.read())  # Use the read method to get the contents as a bytes object
#             f.seek(0)  # Reset the file pointer to the beginning of the file
#             pdf_reader = PdfReader(f)
#             for page in pdf_reader.pages:
#                 raw_text += page.extract_text()
#     return raw_text

def get_pdf_text(pdf):
    raw_text = ""
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(pdf.read())  # Use the read method to get the contents as a bytes object
        f.seek(0)  # Reset the file pointer to the beginning of the file
        pdf_reader = PdfReader(f)
        for page in pdf_reader.pages:
            raw_text += page.extract_text()
    return raw_text

def get_text_chunks(text):
    text_spliter = CharacterTextSplitter(
        separator="\n",
        chunk_size=10000,
        chunk_overlap=1000,
        length_function=len
    )
    chunks = text_spliter.split_text(text)
    return chunks

# Store Chunks to vectore store

def get_vector_store(chunks):
    embeding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
         google_api_key=gemeni_api_key
        )
    # text_list = [str(chunk) for chunk in chunks]  # Convert chunks to a list of strings

    vector_store = FAISS.from_texts(chunks, embeding)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    
    # prompt_template = """
    # Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    # provided context just say, "answer is not available in your provided context", don't provide the wrong answer\n\n
    # Context:\n {context}?\n
    # Question: \n{question}\n

    # Answer:
    # """    
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if you need to more details to perfectly answer the quiestion then ask for more details those you think need to know.
    if the answer is not in provided context just say, "answer is not available in your provided context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=gemeni_api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=['context','question'])

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def handle_user_input(question):
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemeni_api_key)
    saved_faiss_Vbd = FAISS.load_local("faiss_index", embeddings=embedding, allow_dangerous_deserialization=True)
    docs = saved_faiss_Vbd.similarity_search(question)

    chain = get_conversational_chain()
    response = chain(
        {
            "input_documents":docs,
            "question":question,
        },
        return_only_outputs=True
    )

    print(response)
    st.write("Reply: ", response["output_text"])


def main():

    st.set_page_config("Chat with Multiple Pdf")
    st.header("Chat With Multiple pdf using Gemini Pro")

    user_question = st.text_input("Search Your Question in Provided Pdf")

    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your Pdf file and Click on Submit Button")
        if st.button("Submit Pdf's"):
            st.spinner("Processing...")
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Done")



if __name__ == "__main__":
    main()