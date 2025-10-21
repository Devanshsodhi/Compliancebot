# -------------------------------
# Tab 1: Upload Documents (Rewritten)
# -------------------------------
with tab1:
    st.header("📤 Upload and Process Documents")

    # Streamlit form to maintain state
    with st.form("upload_form"):
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True
        )
        submit = st.form_submit_button("🚀 Process Documents")

        if submit:
            if not uploaded_files:
                st.warning("⚠️ Please select at least one PDF file.")
            else:
                # Ensure input directory exists
                input_dir = Path("input")
                input_dir.mkdir(parents=True, exist_ok=True)

                # Save uploaded files
                for file in uploaded_files:
                    with open(input_dir / file.name, "wb") as f:
                        f.write(file.getbuffer())
                st.success(f"✅ {len(uploaded_files)} file(s) saved to `input/`")

                # Initialize Document Processor and Vector Store
                from core.document_processor import DocumentProcessor
                processor = DocumentProcessor()

                # Ensure vector store exists
                if not hasattr(processor, "vector_store") or processor.vector_store is None:
                    from core.vector_store import DocumentVectorStore
                    processor.vector_store = DocumentVectorStore()
                    st.info("ℹ️ Vector store created")

                # Debug: Check ChromaDB client and collection
                st.write("Vector store client:", processor.vector_store.client)
                st.write("Vector store collection:", processor.vector_store.collection)
                st.write("Embedding model loaded:", processor.vector_store.embedding_model is not None)

                # Process PDFs
                st.info("🔄 Processing documents...")
                parsed_docs = processor.process_documents(force_reparse=True, auto_embed=True)

                if parsed_docs:
                    st.success(f"✅ Processed {len(parsed_docs)} document(s)")
                    st.session_state.documents = parsed_docs
                    st.session_state.vector_store = processor.vector_store
                    st.session_state.system_ready = True
                else:
                    st.error("❌ Failed to process documents. Check logs above for details.")
