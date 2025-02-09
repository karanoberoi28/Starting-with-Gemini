# Import Libraries
import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Read Helper Functions
from helper_functions import *

# Get the current working directory
current_directory = os.getcwd()

# Read Google API Key
try:
    # Open the file and read the API key
    with open('google_api_key.txt', 'r') as file:
        google_api_key = file.read().strip()    
    print("API key read successfully!")
    
except FileNotFoundError:
    print("Error: The file google_api_key.txt does not exist. Please check the file path.")

# configure google api key
load_dotenv()
os.environ["google_api_key"] = google_api_key
genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

# Define Parameters
chunk_size = 100000
chunk_overlap = 1000
vector_store = None

# Setup Streamlit
st.set_page_config("Chat PDF")
st.header("Chat with PDF using Gemini")

# Load PDF from sidebar
with st.sidebar:
    st.title("Menu:-")
    pdf_docs = st.file_uploader("Upload PDF File and Click Submit", accept_multiple_files = False)

if pdf_docs:
    with st.spinner("Processing..."):
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text, chunk_size, chunk_overlap)
        vector_store = get_vector_store(text_chunks)
        vector_store.save_local("faiss_index")

# Create conversation with PDF
user_question = st.text_input("Ask a Question from the uploaded file")

if vector_store:

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if user_question:
        response = user_input(user_question, vector_store)["output_text"]
        st.session_state.chat_history.append((user_question, response))        

    for question, answer in st.session_state.chat_history:
        # out_text = '<div class="scrollable-div">' + f"**You:** {question}" + "\n|" + f"**Bot:** {answer}" + '</div>'
        st.markdown(f"**You:** {question}")        
        st.markdown(f"**Bot:** {answer}")