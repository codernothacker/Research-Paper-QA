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
from typing import List, Dict, Any, Optional
import base64

import tornado

# Create temp directory
temp_dir = tempfile.mkdtemp()
atexit.register(lambda: shutil.rmtree(temp_dir))

# Custom CSS
CUSTOM_CSS = """
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
.highlight-controls {
    padding: 10px;
    background-color: #f8f9fa;
    border-radius: 4px;
    margin: 10px 0;
}
.highlighted-text {
    padding: 2px 0;
    border-radius: 2px;
}
.annotation-box {
    background-color: #f8f9fa;
    padding: 10px;
    margin: 5px 0;
    border-left: 3px solid #007bff;
}
.page-link {
    display: inline-block;
    padding: 4px 8px;
    margin: 2px;
    border: 1px solid #ddd;
    border-radius: 4px;
    text-decoration: none;
    color: #0366d6;
}
.page-link:hover {
    background-color: #f0f0f0;
}
.reference-link {
    color: #0366d6;
    text-decoration: none;
    border-bottom: 1px dashed #0366d6;
}
.reference-link:hover {
    border-bottom: 1px solid #0366d6;
}
</style>
"""

@dataclass
class DocumentChunk:
    content: str
    page_number: int
    document_id: str
    document_title: str
    chunk_id: str  # Added for reference tracking

@dataclass
class Highlight:
    text: str
    color: str
    timestamp: datetime
    page_number: Optional[int] = None

@dataclass
class Annotation:
    text: str
    note: str
    timestamp: datetime
    page_number: int
    ref_id: str  # For linking to specific chunks

@dataclass
class Reference:
    source_doc_id: str
    target_doc_id: str
    source_chunk_id: str
    target_chunk_id: str
    reference_text: str
    context: str

class DocumentProcessor:
    @staticmethod
    @staticmethod
    def create_pdf_link(pdf_path: str, filename: str, page_number: Optional[int] = None,
                    chunk_id: Optional[str] = None) -> str:
        try:
            if 'arXiv' in pdf_path:
                arxiv_id = pdf_path.split('arXiv:')[1].split('.pdf')[0]
                base_url = f'https://arxiv.org/pdf/{arxiv_id}.pdf'
                url = f"{base_url}#page={page_number}" if page_number else base_url
                link_class = "page-link" if page_number else "reference-link"
                return f'<a href="{url}" class="{link_class}" target="_blank" data-chunk="{chunk_id}">{filename}</a>'
            
            # Use st.download_button instead of base64 encoding for local files
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
                return st.download_button(
                    label=filename,
                    data=pdf_bytes,
                    file_name=os.path.basename(pdf_path),
                    mime="application/pdf",
                    key=f"download_{chunk_id}_{page_number}"
                )
        except Exception as e:
            st.error(f"Error creating PDF link: {e}")
            return filename  # Return plain text as fallback

    @staticmethod
    def generate_citation(doc_info: Dict[str, Any], style: str = "apa") -> str:
        """Generate citation in different formats"""
        authors = doc_info.get('authors', ['Unknown Author'])
        year = doc_info.get('year', datetime.now().year)
        title = doc_info.get('title', 'Unknown Title')
        
        if style == "apa":
            return f"{', '.join(authors)} ({year}). {title}."
        elif style == "mla":
            return f"{authors[0]}, et al. \"{title}.\" {year}."
        elif style == "chicago":
            return f"{authors[0]}, {title} ({year})."
        return f"{title} - {', '.join(authors)} ({year})"

    @staticmethod
    def highlight_text(text: str, highlights: List[Highlight], chunk_id: Optional[str] = None) -> str:
        """Apply highlighting to text with optional chunk reference"""
        if not highlights:
            return text
        
        highlighted_text = text
        for highlight in highlights:
            highlight_text = highlight.text
            if highlight_text in highlighted_text:
                data_attr = f' data-chunk="{chunk_id}"' if chunk_id else ''
                highlighted_text = highlighted_text.replace(
                    highlight_text,
                    f'<span class="highlighted-text" style="background-color: {highlight.color};"{data_attr}>'
                    f'{highlight_text}</span>'
                )
        return highlighted_text

    @staticmethod
    def extract_document_metadata(file_path: str) -> Dict[str, Any]:
        """Extract metadata from document"""
        try:
            title = os.path.basename(file_path)
            metadata = {
                'title': title,
                'path': file_path,
                'created': datetime.now(),
                'file_type': 'pdf',
                'size': os.path.getsize(file_path)
            }
            
            # Extract arXiv ID if present
            if 'arXiv' in title:
                arxiv_match = re.search(r'arXiv:(\d{4}\.\d{4,5})', title)
                if arxiv_match:
                    metadata['arxiv_id'] = arxiv_match.group(1)
            
            return metadata
        except Exception as e:
            st.error(f"Error extracting metadata: {e}")
            return {'title': os.path.basename(file_path), 'path': file_path}

class QASystem:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.chunk_index = {}  # Map of chunk IDs to their content and metadata

    def initialize(self, chunks: List[DocumentChunk], model_name: str = "llama2") -> bool:
        """Initialize QA system with document chunks"""
        try:
            with st.spinner("Initializing QA System..."):
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )

                # Build chunk index
                self.chunk_index = {
                    chunk.chunk_id: {
                        'content': chunk.content,
                        'page': chunk.page_number,
                        'doc_id': chunk.document_id,
                        'title': chunk.document_title
                    }
                    for chunk in chunks
                }

                documents = [
                    Document(
                        page_content=chunk.content,
                        metadata={
                            'page': chunk.page_number,
                            'document_id': chunk.document_id,
                            'document_title': chunk.document_title,
                            'chunk_id': chunk.chunk_id
                        }
                    )
                    for chunk in chunks
                ]

                self.vectorstore = FAISS.from_documents(documents, self.embeddings)
                llm = Ollama(model=model_name, temperature=0)
                self.qa_chain = ConversationalRetrievalChain.from_llm(
                    llm,
                    self.vectorstore.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True
                )
            return True
        except Exception as e:
            st.error(f"Error initializing QA system: {e}")
            return False

    def semantic_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform semantic search across documents"""
        if not self.vectorstore:
            return []
        return self.vectorstore.similarity_search(query, k=k)

    def ask_question(self, question: str, chat_history: List) -> Dict[str, Any]:
        """Process question and return answer with sources"""
        try:
            if not self.qa_chain:
                st.error("QA system not initialized!")
                return None

            result = self.qa_chain({
                "question": question,
                "chat_history": chat_history
            })

            # Enhance result with chunk references
            for doc in result['source_documents']:
                chunk_id = doc.metadata.get('chunk_id')
                if chunk_id and chunk_id in self.chunk_index:
                    doc.metadata['chunk_data'] = self.chunk_index[chunk_id]

            return result
        except Exception as e:
            st.error(f"Error processing question: {e}")
            return None

class DocumentManager:
    def __init__(self):
        self.primary_doc = None
        self.referenced_docs: Dict[str, Any] = {}
        self.temp_dir = temp_dir
        self.annotations: Dict[str, List[Annotation]] = {}
        self.highlights: Dict[str, List[Highlight]] = {}
        self.references: List[Reference] = []
        self.chunk_counter = 0

    def generate_chunk_id(self) -> str:
        """Generate a unique chunk identifier"""
        self.chunk_counter += 1
        return f"chunk_{self.chunk_counter}"

    def add_highlight(self, doc_id: str, text: str, color: str = "#ffeb3b",
                     page_number: Optional[int] = None) -> None:
        """Add a highlight to the document"""
        if doc_id not in self.highlights:
            self.highlights[doc_id] = []
        self.highlights[doc_id].append(Highlight(text, color, datetime.now(), page_number))

    def add_annotation(self, doc_id: str, text: str, note: str, page_number: int,
                      ref_id: Optional[str] = None) -> None:
        """Add annotation to document"""
        if doc_id not in self.annotations:
            self.annotations[doc_id] = []
        ref_id = ref_id or self.generate_chunk_id()
        self.annotations[doc_id].append(Annotation(text, note, datetime.now(), page_number, ref_id))

    def add_reference(self, source_doc_id: str, target_doc_id: str, 
                     source_chunk_id: str, target_chunk_id: str,
                     reference_text: str, context: str) -> None:
        """Add a reference between documents"""
        self.references.append(Reference(
            source_doc_id, target_doc_id, source_chunk_id,
            target_chunk_id, reference_text, context
        ))

    def extract_references(self, text: str) -> tuple:
        """Extract DOI and arXiv references from text"""
        doi_pattern = r'\b(10\.\d{4,9}/[-._;()/:\w]+)\b'
        arxiv_pattern = r'\b(arXiv:\d{4}\.\d{4,5}(?:v\d+)?)\b'

        dois = list(dict.fromkeys(re.findall(doi_pattern, text, re.IGNORECASE)))
        arxivs = list(dict.fromkeys(re.findall(arxiv_pattern, text, re.IGNORECASE)))
        
        return dois, arxivs

    def find_citation_references(self, text: str) -> List[Dict[str, str]]:
        """Find citation references in text"""
        patterns = [
            r'\[([\d,\s]+)\]',  # [1] or [1,2,3]
            r'\(([^)]+\s*\d{4}[a-z]?(?:,\s*\d{4}[a-z]?)*)\)',  # (Author 2020) or (Author et al. 2020)
        ]
        
        references = []
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                references.append({
                    'text': match.group(0),
                    'content': match.group(1),
                    'start': match.start(),
                    'end': match.end()
                })
        return references

    
    def process_document(self, file_path: str, is_primary: bool = False) -> List[DocumentChunk]:
        """Process document and extract chunks with metadata"""
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
                    chunk_id = self.generate_chunk_id()
                    processed_chunks.append(DocumentChunk(
                        content=chunk.page_content,
                        page_number=chunk.metadata.get('page', 0),
                        document_id=file_path,
                        document_title=title,
                        chunk_id=chunk_id
                    ))

                if is_primary:
                    self.primary_doc = {
                        'path': file_path,
                        'title': title,
                        'chunks': processed_chunks,
                        'metadata': DocumentProcessor.extract_document_metadata(file_path)
                    }

                    # Process references in chunks
                    for chunk in processed_chunks:
                        # Find citations
                        citations = self.find_citation_references(chunk.content)
                        for citation in citations:
                            # Store reference information
                            reference_id = self.generate_chunk_id()
                            self.add_reference(
                                file_path, None, chunk.chunk_id, reference_id,
                                citation['text'], 
                                chunk.content[max(0, citation['start']-50):min(len(chunk.content), citation['end']+50)]
                            )

                    # Extract and process referenced documents
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
                                    'metadata': DocumentProcessor.extract_document_metadata(ref_path)
                                }
                            progress_bar.progress((i + 1) / total_refs)
                        progress_bar.empty()

                return processed_chunks

        except Exception as e:
            st.error(f"Error processing document: {e}")
            return []

    def download_paper(self, identifier: str) -> Optional[str]:
        """Download paper and save to temp directory"""
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

def render_document_section(doc_info: Dict[str, Any], doc_manager: DocumentManager, 
                          doc_processor: DocumentProcessor) -> None:
    """Render a document section with all features"""
    doc_path = doc_info['path']
    
    # Document header with metadata
    st.write(f"### {doc_info['title']}")
    if 'metadata' in doc_info:
        st.markdown("**Document Information:**")
        for key, value in doc_info['metadata'].items():
            if key != 'path':
                st.write(f"- {key.capitalize()}: {value}")

    # Highlighting controls
    with st.container():
        st.markdown('<div class="highlight-controls">', unsafe_allow_html=True)
        text_to_highlight = st.text_area(
            "Text to highlight:",
            key=f"highlight_text_{doc_path}"
        )
        color = st.color_picker(
            "Highlight color:",
            "#ffeb3b",
            key=f"highlight_color_{doc_path}"
        )
        page_num = st.number_input(
            "Page number",
            min_value=1,
            max_value=len(doc_info['chunks']),
            value=1,
            key=f"highlight_page_{doc_path}"
        )
        if st.button("Add Highlight", key=f"highlight_btn_{doc_path}"):
            doc_manager.add_highlight(doc_path, text_to_highlight, color, page_num)
        st.markdown('</div>', unsafe_allow_html=True)

    # Display page navigation
    st.markdown("### Pages")
    page_cols = st.columns(5)
    for i, chunk in enumerate(doc_info['chunks']):
        col_idx = i % 5
        with page_cols[col_idx]:
            pdf_link = doc_processor.create_pdf_link(
                doc_path,
                f"Page {i+1}",
                page_number=i+1,
                chunk_id=chunk.chunk_id
            )
            st.markdown(pdf_link, unsafe_allow_html=True)

    # Display highlighted content with references
    st.markdown("### Content")
    for chunk in doc_info['chunks']:
        with st.container():
            highlighted_text = doc_processor.highlight_text(
                chunk.content,
                doc_manager.highlights.get(doc_path, []),
                chunk.chunk_id
            )
            st.markdown(highlighted_text, unsafe_allow_html=True)

    # Display references
    references = [ref for ref in doc_manager.references 
                 if ref.source_doc_id == doc_path]
    if references:
        st.markdown("### References")
        for ref in references:
            with st.container():
                st.markdown('<div class="reference-box">', unsafe_allow_html=True)
                st.write(f"Reference: {ref.reference_text}")
                st.write("Context:", ref.context)
                if ref.target_doc_id:
                    target_doc = doc_manager.referenced_docs.get(ref.target_doc_id)
                    if target_doc:
                        pdf_link = doc_processor.create_pdf_link(
                            ref.target_doc_id,
                            f"View in {target_doc['title']}",
                            chunk_id=ref.target_chunk_id
                        )
                        st.markdown(pdf_link, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

def initialize_session_state():
    """Initialize all session state variables"""
    if 'doc_manager' not in st.session_state:
        st.session_state.doc_manager = DocumentManager()
    if 'qa_system' not in st.session_state:
        st.session_state.qa_system = QASystem()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    if 'last_search' not in st.session_state:
        st.session_state.last_search = ""
    if 'selected_doc' not in st.session_state:
        st.session_state.selected_doc = None

def handle_file_upload():
    """Handle file upload and initialization"""
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
        return True
    return False

def main():
    """Main application function"""
    try:
        st.title("Research Paper QA System")
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

        # Initialize session state
        initialize_session_state()

        # Add session state for connection status
        if 'connection_active' not in st.session_state:
            st.session_state.connection_active = True
            
        # Handle connection errors
        if not st.session_state.connection_active:
            st.error("Connection lost. Please refresh the page.")
            if st.button("Reconnect"):
                st.session_state.connection_active = True
                st.experimental_rerun()
    except tornado.websocket.WebSocketClosedError:
        st.session_state.connection_active = False
        st.error("Connection lost. Please refresh the page.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

    # Initialize session state
    initialize_session_state()

    # Create document processor
    doc_processor = DocumentProcessor()
    
    # Handle file upload
    if handle_file_upload():
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Q&A", "Documents", "Search", "Annotations"])

        # Q&A Tab
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

                    st.markdown("### Sources:")
                    for i, doc in enumerate(result['source_documents']):
                        with st.container():
                            st.markdown('<div class="source-box">', unsafe_allow_html=True)
                            
                            # Source title with link
                            pdf_link = doc_processor.create_pdf_link(
                                doc.metadata['document_id'],
                                f"Source {i+1}: {doc.metadata['document_title']}",
                                page_number=doc.metadata['page'] + 1,
                                chunk_id=doc.metadata.get('chunk_id')
                            )
                            st.markdown(pdf_link, unsafe_allow_html=True)
                            
                            # Highlighted content with references
                            highlighted_text = doc_processor.highlight_text(
                                doc.page_content,
                                st.session_state.doc_manager.highlights.get(
                                    doc.metadata['document_id'], []
                                ),
                                doc.metadata.get('chunk_id')
                            )
                            st.markdown(highlighted_text, unsafe_allow_html=True)
                            
                            # Citation options
                            citation_style = st.selectbox(
                                "Citation format",
                                ["APA", "MLA", "Chicago"],
                                key=f"citation_{i}"
                            )
                            citation = doc_processor.generate_citation(
                                doc.metadata,
                                style=citation_style.lower()
                            )
                            st.text_area("Citation", citation, key=f"citation_text_{i}")
                            
                            st.markdown('</div>', unsafe_allow_html=True)

                    st.session_state.chat_history.append((question, result['answer']))

        # Documents Tab
        with tab2:
            st.subheader("Document Analysis")
            render_document_section(
                st.session_state.doc_manager.primary_doc,
                st.session_state.doc_manager,
                doc_processor
            )

        # Search Tab
        with tab3:
            st.subheader("Semantic Search")
            search_query = st.text_input("Search documents:", st.session_state.last_search)
            
            if search_query:
                st.session_state.last_search = search_query
                results = st.session_state.qa_system.semantic_search(search_query)
                
                for i, result in enumerate(results):
                    with st.container():
                        st.markdown('<div class="source-box">', unsafe_allow_html=True)
                        
                        # Result header with link
                        pdf_link = doc_processor.create_pdf_link(
                            result.metadata['document_id'],
                            f"Result {i+1} from {result.metadata['document_title']}",
                            page_number=result.metadata['page'] + 1,
                            chunk_id=result.metadata.get('chunk_id')
                        )
                        st.markdown(pdf_link, unsafe_allow_html=True)
                        
                        # Highlighted content with references
                        highlighted_text = doc_processor.highlight_text(
                            result.page_content,
                            st.session_state.doc_manager.highlights.get(
                                result.metadata['document_id'], []
                            ),
                            result.metadata.get('chunk_id')
                        )
                        st.markdown(highlighted_text, unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

        # Annotations Tab
        with tab4:
            st.subheader("Document Annotations & References")
            render_annotations_section(
                st.session_state.doc_manager,
                doc_processor
            )

def render_annotations_section(doc_manager: DocumentManager, doc_processor: DocumentProcessor):
    """Render the annotations section"""
    # Document selector
    available_docs = ([doc_manager.primary_doc['path']] + 
                     list(doc_manager.referenced_docs.keys()))
    doc_id = st.selectbox(
        "Select Document",
        available_docs,
        format_func=lambda x: os.path.basename(x),
        key="annotation_doc_selector"
    )
    
    # Add new annotation
    with st.container():
        st.markdown('<div class="highlight-controls">', unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])
        
        with col1:
            text_to_annotate = st.text_area("Text to annotate:")
            note = st.text_input("Your note:")
        
        with col2:
            page_number = st.number_input(
                "Page number",
                min_value=1,
                max_value=len(doc_manager.primary_doc['chunks']),
                value=1
            )
            if st.button("Add Annotation", key="add_annotation_btn"):
                doc_manager.add_annotation(doc_id, text_to_annotate, note, page_number)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Display existing annotations
    if doc_id in doc_manager.annotations:
        st.markdown("### Existing Annotations")
        for annotation in doc_manager.annotations[doc_id]:
            with st.container():
                st.markdown('<div class="annotation-box">', unsafe_allow_html=True)
                
                # Annotated text with highlights
                highlighted_text = doc_processor.highlight_text(
                    annotation.text,
                    doc_manager.highlights.get(doc_id, []),
                    annotation.ref_id
                )
                st.markdown("**Annotated text:**", unsafe_allow_html=True)
                st.markdown(highlighted_text, unsafe_allow_html=True)
                
                # Note and metadata
                st.markdown("**Note:**", unsafe_allow_html=True)
                st.write(annotation.note)
                st.markdown(f"*Page {annotation.page_number}, Added on {annotation.timestamp.strftime('%Y-%m-%d %H:%M:%S')}*")
                
                # Link to source
                pdf_link = doc_processor.create_pdf_link(
                    doc_id,
                    "View in PDF",
                    page_number=annotation.page_number,
                    chunk_id=annotation.ref_id
                )
                st.markdown(pdf_link, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    # Display references
    refs = [ref for ref in doc_manager.references if ref.source_doc_id == doc_id]
    if refs:
        st.markdown("### Document References")
        for ref in refs:
            with st.container():
                st.markdown('<div class="reference-box">', unsafe_allow_html=True)
                st.write(f"Reference: {ref.reference_text}")
                st.write("Context:", ref.context)
                if ref.target_doc_id:
                    target_doc = doc_manager.referenced_docs.get(ref.target_doc_id)
                    if target_doc:
                        pdf_link = doc_processor.create_pdf_link(
                            ref.target_doc_id,
                            f"View in {target_doc['title']}",
                            chunk_id=ref.target_chunk_id
                        )
                        st.markdown(pdf_link, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()