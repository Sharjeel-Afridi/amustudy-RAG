import os
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
app.config['UPLOAD_FOLDER'] = 'uploads'
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
    
    # Serialize vector index
    vector_index_bytes = pickle.dumps(vector_index)
    
    # Save to PocketBase
    record = {
        "id": file_id,
        "context": context,
        "vector_index": vector_index_bytes
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

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_name = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
        file_id = file_name  # Use the file name as the ID
        file.save(file_path)
        process_pdf(file_path, file_id)
        return jsonify({"filePath": secure_filename(file.filename)})
    return redirect(request.url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/ask', methods=['POST'])
def ask():
    global vector_index, model
    data = request.get_json()
    question = data.get('question', '')
    file_id = data.get('file_id', '')

    if not file_id:
        return jsonify({"response": "No file ID provided."}), 400
    
    if vector_index is None or vector_index.retriever.id != file_id:
        vector_index, context = load_pdf_data(file_id)
    
    if vector_index is None:
        return jsonify({"response": "No PDF file uploaded or processed for this ID."}), 400

    template = """Use the following pieces of context to answer the question but if I ask something else related to this then also answer that. Always be kind after responding.
    {context}
    Question: {question}
    Helpful Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=vector_index,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    if not question:
        return jsonify({"response": "No question provided."}), 400

    result = qa_chain({"query": question})
    return result["result"]

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
