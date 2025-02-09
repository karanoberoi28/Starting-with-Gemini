# Import Libraries
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import google.generativeai as genai

# Read Google API Key
try:
    # Open the file and read the API key
    with open('google_api_key.txt', 'r') as file:
        google_api_key = file.read().strip()    
    print("API key read successfully!")
    genai.configure(api_key = google_api_key)
    
except FileNotFoundError:
    print("Error: The file google_api_key.txt does not exist. Please check the file path.")


# Helper Functions
def get_pdf_text(pdf):
    text=""
    pdf_reader= PdfReader(pdf)
    for page in pdf_reader.pages:
        text+= page.extract_text()
    return  text

def get_text_chunks(text, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store
    # vector_store.save_local("faiss_index")

def get_conversational_chat():

    prompt_template = """
    Answer the question as detailed as possible from the provided context. Ensure to provide all the details. Don't provide wrong answers\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt = prompt)

    return chain

def user_input(user_question, vector_store):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chat()

    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs = True)

    return response