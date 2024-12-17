from flask import Flask, request, render_template, send_file, jsonify,send_from_directory
import tempfile
import shutil
import os
import re
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from dataclasses import dataclass
from typing import List
import atexit

app = Flask(__name__)

# Create temp directory
temp_dir = tempfile.mkdtemp()

atexit.register(lambda: shutil.rmtree(temp_dir))
def initialize_temp_folder():
    """Initialize temp folder."""
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

# Initialize the temp folder manually before handling requests
initialize_temp_folder()

def setup_cleanup():
    """Ensure cleanup of temp directory on exit."""
    import atexit
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
            return str(e)

    def ask_question(self, question: str, chat_history: List):
        """Process question and return answer with sources."""
        try:
            if not self.qa_chain:
                return None, "QA system not initialized!"
            result = self.qa_chain({
                "question": question,
                "chat_history": chat_history
            })
            return result, None
        except Exception as e:
            return None, str(e)


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
            return []


doc_manager = DocumentManager()
qa_system = QASystem()
chat_history = []


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file:
        file_path = os.path.join(temp_dir, "uploaded.pdf")
        file.save(file_path)
        doc_manager.process_document(file_path, is_primary=True)
        return jsonify({"message": "File uploaded and processed successfully!"})
    return jsonify({"error": "No file uploaded!"}), 400


@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '')
    if question:
        result, error = qa_system.ask_question(question, chat_history)
        if error:
            return jsonify({"error": error}), 500
        return jsonify({
            "answer": result['answer'],
            "sources": [{
                "content": doc.page_content,
                "title": doc.metadata['document_title'],
                "page": doc.metadata['page']
            } for doc in result['source_documents']]
        })
    return jsonify({"error": "No question provided!"}), 400


@app.route('/preview/<filename>')
def preview(filename):
    file_path = os.path.join(temp_dir, filename)
    if os.path.exists(file_path):
        return send_file(file_path)
    return jsonify({"error": "File not found!"}), 404


if __name__ == '__main__':
    app.run(debug=True)
