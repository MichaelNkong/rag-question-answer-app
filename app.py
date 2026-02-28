import os
import streamlit as st
from rag_utility import process_document_to_chroma_db, answer_question

working_dir = os.getcwd()

st.title("📄 Document Q&A RAG App")
st.subheader("Upload PDFs and ask questions about their content")
st.caption("Powered by LangChain, Chroma, and HuggingFace embeddings")

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
        with st.spinner("Processing PDFs and creating embeddings..."):
            if process_document_to_chroma_db(save_path):
                  st.success("Documents processed successfully ✅")
            else:
                  st.error(f"Error processing document:")


# Ask question
user_question = st.text_area("Ask your question about the document", height=150, key="user_question")
if st.button("Get Answer", type="primary"):
    if "vectorDB" not in st.session_state:
        st.warning("Please upload and process at least one document first.")
    elif not user_question.strip():
        st.warning("Please enter a question.")
    else:
        answer, sources = answer_question(user_question)

        with st.expander("View Answer"):
            st.markdown(answer)

        if sources:
            with st.expander("View Source Documents"):
                for doc in sources:
                    st.markdown(f"- {doc.metadata['source']}")