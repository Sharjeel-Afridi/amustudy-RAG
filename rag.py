import os
from pathlib import Path
import warnings
import pickle
from flask import Flask, request, jsonify, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from flask_cors import CORS
from pocketbase import PocketBase

load_dotenv()
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'documents'
CORS(app)

# PocketBase setup
pb = PocketBase("https://amustudy-rag.pockethost.io/")

model = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=os.getenv('GOOGLE_API_KEY'),
    temperature=0.2,
    convert_system_message_to_human=True
)




def process_pdf(file_path, file_id):
    global vector_index
    
    pdf_loader = PyPDFLoader(file_path)
    pages = pdf_loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    context = "\n\n".join(str(p.page_content) for p in pages)
    texts = text_splitter.split_text(context)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv('GOOGLE_API_KEY')
    )

    vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 6})
    
    # Extract data from vector index
    index_data = {
        'texts': texts,  # This should be serialized to a format that can be saved and restored
        # 'embeddings': embeddings.__dict__,  # Save relevant attributes if needed
        'context': context,
    }

    # Convert to JSON-serializable format
    index_data_json = {
        'texts': index_data['texts'],
        'context': index_data['context'],
    }

    # Save to PocketBase
    record = {
        # "id": file_id,
        # "context": context,
        "vector_index": index_data_json
    }
    pb.collection("pdfs").create(record)

def load_pdf_data(file_id):
    global vector_index
    record = pb.collection("pdfs").get_one(file_id)
    if record:
        vector_index = pickle.loads(record["vector_index"])
        context = record["context"]
        return vector_index, context
    return None, None


def process_all_pdfs():
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if filename.endswith('.pdf'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file_id = Path(filename).stem  # Use the file name without extension as the ID
            process_pdf(file_path, file_id)


if __name__ == "__main__":
    # os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    process_all_pdfs()  
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
