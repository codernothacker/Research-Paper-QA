import streamlit as st
import tempfile
import shutil
import atexit
import requests
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from dataclasses import dataclass
from typing import List
import re

# Create temp directory
temp_dir = tempfile.mkdtemp()
atexit.register(lambda: shutil.rmtree(temp_dir))

@dataclass
class DocumentChunk:
    content: str
    page_number: int
    document_id: str
    document_title: str

class QASystem:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        
    def initialize(self, chunks: List[DocumentChunk], model_name: str = "llama2"):
        """Initialize QA system with document chunks."""
        try:
            # Create embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            # Convert chunks to LangChain documents
            documents = [
                Document(
                    page_content=chunk.content,
                    metadata={
                        'page': chunk.page_number,
                        'document_id': chunk.document_id,
                        'document_title': chunk.document_title
                    }
                )
                for chunk in chunks
            ]
            
            # Create vectorstore
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            
            # Setup QA chain
            llm = Ollama(
                model=model_name,
                temperature=0
            )
            
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm,
                self.vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )
            
            return True
        except Exception as e:
            st.error(f"Error initializing QA system: {e}")
            return False

    def ask_question(self, question: str, chat_history: List):
        """Process question and return answer with sources."""
        try:
            if not self.qa_chain:
                st.error("QA system not initialized!")
                return None
                
            result = self.qa_chain({
                "question": question,
                "chat_history": chat_history
            })
            return result
        except Exception as e:
            st.error(f"Error processing question: {e}")
            return None

class DocumentManager:
    def __init__(self):
        self.primary_doc = None
        self.referenced_docs = {}
        self.temp_dir = temp_dir
        
    def extract_references(self, text):
        """Extract DOI and arXiv references from text."""
        doi_pattern = r'10.\\d{4,9}/[-._;()/:\\w]+'
        arxiv_pattern = r'arXiv:\\d{4}.\\d{4,5}'
        
        dois = re.findall(doi_pattern, text)
        arxivs = re.findall(arxiv_pattern, text)
        return dois, arxivs
    
    def download_paper(self, identifier):
        """Download paper and save to temp directory."""
        try:
            if identifier.startswith('10.'):
                response = requests.get(f'https://doi.org/{identifier}',
                                     headers={'Accept': 'application/pdf'})
            else:
                arxiv_id = identifier.split(':')[1]
                response = requests.get(f'https://arxiv.org/pdf/{arxiv_id}.pdf')
            
            if response.status_code == 200:
                filename = os.path.join(self.temp_dir, f"paper_{identifier.replace('/', '_')}.pdf")
                with open(filename, 'wb') as f:
                    f.write(response.content)
                return filename
            return None
        except Exception as e:
            st.error(f"Error downloading paper {identifier}: {e}")
            return None

    def process_document(self, file_path, is_primary=False):
        """Process document and extract chunks with metadata."""
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(pages)
            
            # Extract document title
            title = os.path.basename(file_path)
            
            processed_chunks = []
            for chunk in chunks:
                processed_chunks.append(DocumentChunk(
                    content=chunk.page_content,
                    page_number=chunk.metadata.get('page', 0),
                    document_id=file_path,
                    document_title=title
                ))
            
            if is_primary:
                self.primary_doc = {
                    'path': file_path,
                    'title': title,
                    'chunks': processed_chunks
                }
                
                # Process references
                full_text = ' '.join([chunk.content for chunk in processed_chunks])
                dois, arxivs = self.extract_references(full_text)
                
                with st.spinner(f"Found {len(dois) + len(arxivs)} references. Downloading..."):
                    for identifier in dois + arxivs:
                        ref_path = self.download_paper(identifier)
                        if ref_path:
                            ref_chunks = self.process_document(ref_path)
                            self.referenced_docs[ref_path] = {
                                'path': ref_path,
                                'title': title,
                                'chunks': ref_chunks
                            }
            
            return processed_chunks
            
        except Exception as e:
            st.error(f"Error processing document: {e}")
            return []

def main():
    st.title("Research Paper QA System")
    
    # Initialize session state
    if 'doc_manager' not in st.session_state:
        st.session_state.doc_manager = DocumentManager()
    if 'qa_system' not in st.session_state:
        st.session_state.qa_system = QASystem()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # File upload
    uploaded_file = st.file_uploader("Upload Research Paper (PDF)", type="pdf")
    
    if uploaded_file:
        # Save primary document
        primary_path = os.path.join(temp_dir, "primary.pdf")
        with open(primary_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Process documents if not already processed
        if not st.session_state.doc_manager.primary_doc:
            with st.spinner("Processing primary document and downloading references..."):
                st.session_state.doc_manager.process_document(primary_path, is_primary=True)
            
            # Initialize QA system
            all_chunks = (
                st.session_state.doc_manager.primary_doc['chunks'] +
                [chunk for doc in st.session_state.doc_manager.referenced_docs.values() 
                 for chunk in doc['chunks']]
            )
            st.session_state.qa_system.initialize(all_chunks)
        
        # Q&A Tab
        st.subheader("Ask Questions")
        question = st.text_input("Enter your question:")
        
        if question:
            # Get answer
            result = st.session_state.qa_system.ask_question(
                question,
                st.session_state.chat_history
            )
            
            if result:
                # Display answer
                st.markdown("### Answer:")
                st.write(result['answer'])
                
                # Display sources
                st.markdown("### Sources Used:")
                for i, doc in enumerate(result['source_documents']):
                    with st.expander(f"Source {i+1} from {doc.metadata['document_title']}"):
                        st.write(doc.page_content)
                
                # Update chat history
                st.session_state.chat_history.append((question, result['answer']))

if __name__ == "__main__":
    main()
