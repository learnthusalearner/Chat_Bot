import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from htmltemplate import css, bot_template, user_template

VECTORSTORE_DIR = "vectorstore_index"


@st.cache_data
def get_pdf_chunks_with_metadata(pdf_docs):
    chunks_with_metadata = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                metadata = {"source": pdf.name, "page": page_num + 1}
                chunks_with_metadata.append((text, metadata))
    return chunks_with_metadata

@st.cache_data
def split_chunks_with_metadata(chunks_with_metadata):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    texts, metadatas = [], []
    for text, metadata in chunks_with_metadata:
        for chunk in text_splitter.split_text(text):
            texts.append(chunk)
            metadatas.append(metadata)
    return texts, metadatas

@st.cache_resource
def get_vectorstore(text_chunks, metadatas):
    embeddings = OllamaEmbeddings(model="all-minilm")
    if os.path.exists(VECTORSTORE_DIR):
        return FAISS.load_local(
            VECTORSTORE_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        metadatas=metadatas
    )
    vectorstore.save_local(VECTORSTORE_DIR)
    return vectorstore


@st.cache_resource
def get_conversation_chain(_vectorstore):
    llm = ChatOllama(model="llama3", temperature=0.5)
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=False,
        output_key="answer"
    )


def handle_userinput(user_question):
    result = st.session_state.conversation({ "question": user_question })
    answer = result["answer"]

    
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []
    st.session_state.qa_history.append((user_question, answer))


    st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)

    # Source of page from the db 
    docs = st.session_state.conversation.retriever.get_relevant_documents(user_question)
    for doc in docs:
        meta = doc.metadata
        st.markdown(
            f"<div style='font-size: small; color: gray;'>"
            f"Source: {meta.get('source','Unknown')} (Page {meta.get('page','?')})"
            f"</div>",
            unsafe_allow_html=True
        )


def main():
    load_dotenv()
    st.set_page_config(page_title="IIT Kanpur | IT Helpdesk Assistant", page_icon=None)
    st.write(css, unsafe_allow_html=True)

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.title("IIT Kanpur – IT Helpdesk Assistant")
    st.markdown(
        "This assistant can answer queries based on uploaded IT/networking documents. "
        "Use the sidebar to upload PDF documents and then ask your questions below."
    )

    # Text input
    user_question = st.text_input("Enter your question:")
    if user_question and st.session_state.conversation:
        handle_userinput(user_question)

    # Sidebar: Upload + History
    with st.sidebar:
        st.header("Upload IT Documents")
        pdf_docs = st.file_uploader(
            "Upload one or more PDF files", 
            accept_multiple_files=True,
        )

        if st.button("Process") and pdf_docs:
            with st.spinner("Processing documents..."):
                chunks_meta = get_pdf_chunks_with_metadata(pdf_docs)
                texts, metas = split_chunks_with_metadata(chunks_meta)
                vectorstore = get_vectorstore(texts, metas)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.session_state.qa_history = []  # reseting chat history
            st.success("Documents processed successfully. You can now ask your questions.")

        # Show history
        if "qa_history" in st.session_state and st.session_state.qa_history:
            st.divider()
            st.subheader("Chat History")
            for i, (q, a) in enumerate(reversed(st.session_state.qa_history), 1):
                with st.expander(f"Q{i}: {q}"):
                    st.markdown(a)

if __name__ == '__main__':
    main()
