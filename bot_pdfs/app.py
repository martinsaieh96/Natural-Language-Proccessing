import streamlit as st
from dotenv import load_dotenv
from bot_pdfs.utils import *

def main():
    load_dotenv()
    st.set_page_config(page_title="Chatea con tus Documentos",
                       page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chatea con tus Documentos :books:")
    user_question = st.text_input("Haz una pregunta a tus documentos:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Tus documentos")
        pdf_docs = st.file_uploader(
            "Carga aquí tus archivos PDF'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Procesando"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()