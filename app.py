import os
import pytesseract
from docx2python import docx2python
from PIL import Image
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
import streamlit as st
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.llms import HuggingFaceHub
from PyPDF2 import PdfReader

if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
    # Explicitely set the api token
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""

def extract_text_from_images(image_paths):
    texts = []
    img = Image.open(image_paths)
    img = img.convert('RGBA')
    texts.append(pytesseract.image_to_string(img, lang='eng', config='--psm 6'))
    return texts

def extract_text_from_pdfs(pdf_paths):
    texts = []
    with open(pdf_paths, 'rb') as f:
        pdf = PdfReader(f)
        for page_num in range(len(pdf.pages)):
            page = pdf.pages[page_num]
            texts.append(page.extract_text())
    return texts
    
def extract_text_from_docx(docx_paths):
    texts = []
    with docx2python(docx_paths) as docx_content:
        texts.append(docx_content.text)
    return texts

def create_vectorstore(embeddings):
    raw_data = []
    for element in os.listdir("data"):
        full_path = os.path.join("data", element)
        if element.endswith("pdf") or element.endswith("PDF"):
            raw_data.extend(extract_text_from_pdfs(full_path))
        if element.endswith("docx") or element.endswith("DOCX"):
            raw_data.extend(extract_text_from_docx(full_path))
        else:
            raw_data.extend(extract_text_from_images(full_path))

    text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=0)
    texts = text_splitter.create_documents(raw_data)
    docsearch = FAISS.from_documents(texts, embeddings)
    
    return docsearch

def get_conversation(vectorstore, model):
    memory = ConversationBufferMemory(memory_key="messages", return_messages=True)
    conversation_chain = RetrievalQA.from_llm(
        llm=model,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def get_response(conversation_chain, query):
    response = conversation_chain.invoke(query)
    answer = response["result"].split('\nHelpful Answer: ')[1]
    return answer

def main():
    st.title("DocChat LLM")
    if not os.path.exists("data"):
        os.makedirs("data")
    
    with st.spinner('Downloading Model ...'):
        embeddings = HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-base",
            model_kwargs={'device': 'cpu'}
        )
    
    with st.spinner('Loading LLM ...'):
        llm = HuggingFaceHub(
            repo_id="HuggingFaceH4/zephyr-7b-beta", 
            model_kwargs={
                "temperature": 0.7, 
                "input_tokens": 512, 
                "max_new_tokens": 1024, 
                "top_p": 0.95, 
                "top_k": 50
            }
        )
    
    st.sidebar.title("Upload Document(s)")
    uploaded_files = st.sidebar.file_uploader("Upload Documents", accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            with open(f"data/{file.name}", "wb") as f:
                f.write(file.getbuffer())
        with st.spinner('Creating database ...'):
            vectorstore =     create_vectorstore(embeddings)
        with st.spinner('Creating chain ...'):
            conversation_chain = get_conversation(vectorstore, llm)
        st.sidebar.success("Document(s) uploaded successfully")
    else:
        st.sidebar.warning("Require at least one document")
    
    if st.sidebar.button("Clear"):
        st.session_state.messages = []
        filelist = [ f for f in os.listdir("data")]
        for f in filelist:
            os.remove(os.path.join("data", f))
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        if message["role"] == "human":
            st.chat_message("human").markdown(message["content"])
        else:
            st.chat_message("assistant").markdown(message["content"])
    
    user_prompt = st.chat_input("Question", key="human")
    if user_prompt:
        st.chat_message("human").markdown(user_prompt)
        st.session_state.messages.append({"role": "human", "content": user_prompt})
        response = get_response(conversation_chain, user_prompt)
        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
