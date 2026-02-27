import os
import streamlit as st
from rag_utility import process_document_to_chroma_db, answer_question

working_dir = os.getcwd()

st.title("Document RAG")

# Allow multiple PDF uploads
uploaded_files = st.file_uploader(
    "Upload PDF(s)", type=["pdf"], accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        save_path = os.path.join(working_dir, uploaded_file.name)
        # Save file to local working directory
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process PDF to Chroma vector store
        process_document_to_chroma_db(uploaded_file.name)

    st.success("All documents processed successfully!")

# Ask question
user_question = st.text_area("Ask your question about the documents")
if st.button("Answer"):
    if "vectorDB" not in st.session_state:
        st.warning("Please upload and process at least one document first.")
    elif not user_question.strip():
        st.warning("Please enter a question.")
    else:
        answer, sources = answer_question(user_question)

        st.markdown("### 📌 Response")
        st.markdown(answer)

        if sources:
            st.markdown("### 📄 Sources")
            for doc in sources:
                st.markdown(f"- {doc.page_content[:200]}...")