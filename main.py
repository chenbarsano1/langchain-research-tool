import os
import streamlit as st
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

llm = GoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.6)

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()
save_path = "faiss_index"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if process_url_clicked:
    valid_urls = [url for url in urls if url.strip()]

    if not valid_urls:
        st.error("Please enter at least one URL!")
    
    else:
        try:
            loader = SeleniumURLLoader(urls=valid_urls)
            main_placeholder.text("Data Loading...Started...✅✅✅")
            data = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1500,
                chunk_overlap=150
            )
            main_placeholder.text("Text Splitter...Started...✅✅✅")
            docs = text_splitter.split_documents(data)

            vector_store = FAISS.from_documents(docs, embeddings)
            main_placeholder.text("Embedding Vector Started Building...✅✅✅")

            vector_store.save_local(save_path)
            main_placeholder.empty()
            st.success("✅ Processing complete! You can now ask questions.")

        except Exception as e:
            st.error(f"Error processing URLs: {str(e)}")
            import traceback
            st.error(traceback.format_exc())


query = st.text_input("Question: ")
if query:
    if os.path.exists(save_path):
        try:
            vector_store = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
            retriever = vector_store.as_retriever(search_kwargs={"k": 8})

            # Changed this line - use invoke() instead of get_relevant_documents()
            docs = retriever.invoke(query)

            # Combine context
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Create prompt
            prompt = f"""Answer the question based only on the following context. Be specific and accurate.
If you cannot find the answer in the context, say "I cannot find this information in the provided sources."

Context: {context}

Question: {query}

Answer:"""
            
            # Call Gemini
            with st.spinner("Thinking..."):
                answer = llm.invoke(prompt)

            # Display answer
            st.header("Answer")
            st.write(answer)

            # Display sources
            st.subheader("Sources:")
            sources = set()
            for doc in docs:
                if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                    sources.add(doc.metadata['source'])
            
            for source in sources:
                st.write(f"- {source}")
            
            # Show context for debugging
            with st.expander("📄 View Retrieved Context"):
                for i, doc in enumerate(docs):
                    st.write(f"**Chunk {i+1}:**")
                    st.write(doc.page_content[:400] + "...")
                    st.write("---")

        except Exception as e:
            st.error(f"Error answering question: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

    else:
        st.warning("Please process URLs first before asking questions!")