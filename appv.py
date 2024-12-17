import streamlit as st
import tempfile
import shutil
import atexit
import requests
import os
import re
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from dataclasses import dataclass
from typing import List
import base64

# Create temp directory
temp_dir = tempfile.mkdtemp()
atexit.register(lambda: shutil.rmtree(temp_dir))

@dataclass
class DocumentChunk:
    content: str
    page_number: int
    document_id: str
    document_title: str

def create_pdf_link(pdf_path, filename):
    """Create a downloadable/viewable link for PDF files"""
    try:
        if 'arXiv' in pdf_path:
            arxiv_id = pdf_path.split('arXiv:')[1].split('.pdf')[0]
            href = f'<a href="https://arxiv.org/pdf/{arxiv_id}.pdf" target="_blank">{filename}</a>'
            return href
        
        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        
        href = f'''
            <a href="data:application/pdf;base64,{base64_pdf}" 
               target="_blank" 
               style="text-decoration: none; color: #0366d6; cursor: pointer;">
                {filename}
            </a>
        '''
        return href
    except Exception as e:
        st.error(f"Error creating PDF link: {e}")
        return None

def generate_citation(doc_info, style="apa"):
    """Generate citation in different formats"""
    authors = doc_info.get('authors', ['Unknown Author'])
    year = doc_info.get('year', datetime.now().year)
    title = doc_info.get('title', 'Unknown Title')
    
    if style == "apa":
        return f"{', '.join(authors)} ({year}). {title}."
    elif style == "mla":
        return f"{authors[0]}, et al. \"{title}.\" {year}."
    return f"{title} - {', '.join(authors)} ({year})"

class QASystem:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None

    def initialize(self, chunks: List[DocumentChunk], model_name: str = "llama2"):
        """Initialize QA system with document chunks."""
        try:
            with st.spinner("Initializing QA System..."):
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )

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

                self.vectorstore = FAISS.from_documents(documents, self.embeddings)

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

    def semantic_search(self, query: str, k: int = 5):
        """Perform semantic search across documents"""
        if not self.vectorstore:
            return []
        return self.vectorstore.similarity_search(query, k=k)

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
        self.annotations = {}
        self.highlights = {}  # New dictionary to store highlights

    def add_highlight(self, doc_id, text, color="#ffeb3b"):
        """Add a highlight to the document"""
        if doc_id not in self.highlights:
            self.highlights[doc_id] = []
        self.highlights[doc_id].append({
            "text": text,
            "color": color,
            "timestamp": datetime.now()
        })
    def extract_references(self, text):
        """Extract DOI and arXiv references from text."""
        doi_pattern = r'\b(10\.\d{4,9}/[-._;()/:\w]+)\b'
        arxiv_pattern = r'\b(arXiv:\d{4}\.\d{4,5}(?:v\d+)?)\b'

        dois = re.findall(doi_pattern, text, re.IGNORECASE)
        arxivs = re.findall(arxiv_pattern, text, re.IGNORECASE)
        
        dois = list(dict.fromkeys(dois))
        arxivs = list(dict.fromkeys(arxivs))
        
        return dois, arxivs

    def add_annotation(self, doc_id, text, note):
        """Add annotation to document"""
        if doc_id not in self.annotations:
            self.annotations[doc_id] = []
        self.annotations[doc_id].append({
            "text": text,
            "note": note,
            "timestamp": datetime.now()
        })

    def generate_summary(self, chunks):
        """Generate summary from document chunks"""
        full_text = " ".join([chunk.content for chunk in chunks[:3]])
        return full_text[:500] + "..."

    def download_paper(self, identifier):
        """Download paper and save to temp directory."""
        try:
            if identifier.startswith('10.'):
                response = requests.get(
                    f'https://doi.org/{identifier}',
                    headers={'Accept': 'application/pdf'},
                    allow_redirects=True,
                    timeout=30
                )
                filename = f"paper_doi_{identifier.replace('/', '_')}.pdf"
            else:
                arxiv_id = identifier.split(':')[1]
                response = requests.get(
                    f'https://arxiv.org/pdf/{arxiv_id}.pdf',
                    timeout=30
                )
                filename = f"paper_arXiv:{arxiv_id}.pdf"

            if response.status_code == 200 and 'application/pdf' in response.headers.get('content-type', '').lower():
                filepath = os.path.join(self.temp_dir, filename)
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                return filepath
            return None
        except Exception as e:
            st.warning(f"Could not download paper {identifier}: {str(e)}")
            return None

    def process_document(self, file_path, is_primary=False):
        """Process document and extract chunks with metadata."""
        try:
            with st.spinner(f"Processing {'primary' if is_primary else 'referenced'} document..."):
                loader = PyPDFLoader(file_path)
                pages = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = text_splitter.split_documents(pages)

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
                        'chunks': processed_chunks,
                        'summary': self.generate_summary(processed_chunks)
                    }

                    full_text = ' '.join([chunk.content for chunk in processed_chunks])
                    dois, arxivs = self.extract_references(full_text)
                    
                    total_refs = len(dois) + len(arxivs)
                    if total_refs > 0:
                        progress_bar = st.progress(0)
                        for i, identifier in enumerate(dois + arxivs):
                            ref_path = self.download_paper(identifier)
                            if ref_path:
                                ref_chunks = self.process_document(ref_path)
                                self.referenced_docs[ref_path] = {
                                    'path': ref_path,
                                    'title': os.path.basename(ref_path),
                                    'chunks': ref_chunks,
                                    'summary': self.generate_summary(ref_chunks)
                                }
                            progress_bar.progress((i + 1) / total_refs)
                        progress_bar.empty()

                return processed_chunks

        except Exception as e:
            st.error(f"Error processing document: {e}")
            return []

def main():
    st.title("Research Paper QA System")
    st.markdown("""
        <style>
        .source-box {
            border: 1px solid #ddd;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
            background-color: #ffffff;
        }
        .source-link {
            margin-top: 10px;
            margin-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    if 'doc_manager' not in st.session_state:
        st.session_state.doc_manager = DocumentManager()
    if 'qa_system' not in st.session_state:
        st.session_state.qa_system = QASystem()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'annotations' not in st.session_state:
        st.session_state.annotations = {}

    uploaded_file = st.file_uploader("Upload Research Paper (PDF)", type="pdf")

    if uploaded_file:
        primary_path = os.path.join(temp_dir, "primary.pdf")
        with open(primary_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        if not st.session_state.doc_manager.primary_doc:
            st.session_state.doc_manager.process_document(primary_path, is_primary=True)

            all_chunks = (
                st.session_state.doc_manager.primary_doc['chunks'] +
                [chunk for doc in st.session_state.doc_manager.referenced_docs.values()
                 for chunk in doc['chunks']]
            )
            st.session_state.qa_system.initialize(all_chunks)

        tab1, tab2, tab3, tab4 = st.tabs(["Q&A", "Documents", "Search", "Annotations"])

        with tab1:
            st.subheader("Ask Questions")
            question = st.text_input("Enter your question:")

            if question:
                result = st.session_state.qa_system.ask_question(
                    question,
                    st.session_state.chat_history
                )

                if result:
                    st.markdown("### Answer:")
                    st.write(result['answer'])

                    st.markdown("### Sources Used:")
                    for i, doc in enumerate(result['source_documents']):
                        with st.container():
                            st.markdown(f'<div class="source-box">', unsafe_allow_html=True)
                            st.write(f"Source {i+1} from {doc.metadata['document_title']}")
                            st.write(doc.page_content)
                            
                            st.markdown('<div class="source-link">', unsafe_allow_html=True)
                            pdf_link = create_pdf_link(
                                doc.metadata['document_id'],
                                f"View {doc.metadata['document_title']}"
                            )
                            st.markdown(pdf_link, unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            citation_style = st.selectbox(
                                "Citation format",
                                ["APA", "MLA"],
                                key=f"citation_{i}"
                            )
                            
                            filename = doc.metadata['document_title']
                            if 'arXiv:' in filename:
                                arxiv_id = filename.split('arXiv:')[1].split('.pdf')[0]
                                citation = f"arXiv:{arxiv_id}" if citation_style == "APA" else f"arXiv preprint arXiv:{arxiv_id}"
                            else:
                                citation = generate_citation(
                                    {"title": doc.metadata['document_title']},
                                    style=citation_style.lower()
                                )
                            
                            st.text_area("Citation", citation, key=f"citation_text_{i}")
                            st.markdown('</div>', unsafe_allow_html=True)

                    st.session_state.chat_history.append((question, result['answer']))

        with tab2:
            st.subheader("Document Summaries")
            
            st.write("### Primary Document")
            primary_doc = st.session_state.doc_manager.primary_doc
            if primary_doc:
                st.write(primary_doc['title'])
                st.write("Summary:")
                st.write(primary_doc['summary'])
                pdf_link = create_pdf_link(primary_doc['path'], "View PDF")
                st.markdown(pdf_link, unsafe_allow_html=True)

            st.write("### Referenced Documents")
            for doc_path, doc_info in st.session_state.doc_manager.referenced_docs.items():
                with st.expander(doc_info['title']):
                    st.write("Summary:")
                    st.write(doc_info['summary'])
                    pdf_link = create_pdf_link(doc_path, "View PDF")
                    st.markdown(pdf_link, unsafe_allow_html=True)

        with tab3:
            st.subheader("Semantic Search")
            search_query = st.text_input("Enter search term:")
            if search_query:
                results = st.session_state.qa_system.semantic_search(search_query)
                for i, result in enumerate(results):
                    with st.container():
                        st.markdown(f'<div class="source-box">', unsafe_allow_html=True)
                        st.write(f"Result {i+1} from {result.metadata['document_title']}")
                        st.write(result.page_content)
                        pdf_link = create_pdf_link(
                            result.metadata['document_id'],
                            f"View {result.metadata['document_title']}"
                        )
                        st.markdown(pdf_link, unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                # Annotations Tab
        with tab4:
            st.subheader("Document Annotations")
            doc_id = st.selectbox(
                "Select Document",
                [primary_doc['path']] + list(st.session_state.doc_manager.referenced_docs.keys()),
                format_func=lambda x: os.path.basename(x)
            )
            
            text_to_annotate = st.text_area("Text to annotate:")
            note = st.text_input("Your note:")
            if st.button("Add Annotation"):
                st.session_state.doc_manager.add_annotation(doc_id, text_to_annotate, note)

            # Display annotations
            if doc_id in st.session_state.doc_manager.annotations:
                for annotation in st.session_state.doc_manager.annotations[doc_id]:
                    with st.container():
                        st.markdown(f'<div class="source-box">', unsafe_allow_html=True)
                        st.write("Original text:", annotation["text"])
                        st.write("Note:", annotation["note"])
                        st.write("Time:", annotation["timestamp"].strftime("%Y-%m-%d %H:%M:%S"))
                        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()