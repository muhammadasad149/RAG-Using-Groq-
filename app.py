import streamlit as st
from groq import Groq
import random
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
import tempfile
from dotenv import load_dotenv
load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.environ["GOOGLE_API_KEY"])
groq_api_key = os.environ['groq_api_key']

def get_vectorstore_from_pdf(pdf_file):
    # Save the uploaded PDF file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        pdf_file_path = tmp_file.name
    
    # get the text in document form
    loader = PyPDFLoader(pdf_file_path)    
    document = loader.load()
    
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    
    # create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, embeddings)

    # Delete the temporary file after use
    os.unlink(pdf_file_path)

    return vector_store

def get_context_retriever_chain(llm,vector_store):
    llm = groq_chat
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    
def get_conversational_rag_chain(llm,retriever_chain): 
    
    llm = groq_chat
    
    prompt = ChatPromptTemplate.from_messages([
       #"Answer the user's questions based on the below context:\n\n{context}"
      ("system",  """Answer the user's questions based on the below context provide a relevant and concise answer based on the provided context. If the answer is not present in the context, state "I don't know.
        CONTEXT: {context}

        Input: {input}"""),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)



def get_response(llm, user_input):
    retriever_chain = get_context_retriever_chain(llm, st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(llm, retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']


# app config
st.set_page_config(page_title="Chat with PDF Using Groq", page_icon="🤖")
st.title("Chat with PDF")


# sidebar
st.sidebar.title('select the LLM')
model = st.sidebar.selectbox(
        'Choose a model',
        ['mixtral-8x7b-32768', 'llama2-70b-4096']
    )

groq_chat = ChatGroq(
        groq_api_key = groq_api_key,
        model = model
    )

with st.sidebar:
    st.header("upload pdf")
    pdf_file = st.sidebar.file_uploader("Upload PDF", type="pdf")

if pdf_file is None or pdf_file == "":
    st.info("Please upload the pdf file")

else:
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_pdf(pdf_file)    

    # user input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        response = get_response(groq_chat,user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        
       

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
