from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from langchain.vectorstores import Pinecone as PineconeLangChain
from langchain.embeddings import HuggingFaceEmbeddings
import pinecone
import os
import uvicorn

app = FastAPI()

# Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))
index_name = "hackrx-index"

# Initialize Embeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

# Initialize Vector Store
vector_store = PineconeLangChain.from_existing_index(index_name, embeddings)

# Input Schema
class RunRequest(BaseModel):
    documents: str
    questions: list[str]

# Download PDF and save locally
def download_pdf(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
    else:
        raise HTTPException(status_code=400, detail="Failed to download document.")

# Process Document and Update Vector DB
def process_document(file_path):
    from langchain.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    loader = PyPDFLoader(file_path)
    raw_documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(raw_documents)
    vector_store.add_documents(chunks)

# HuggingFace Inference API Call
def call_huggingface_inference(prompt):
    api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}

    payload = {
        "inputs": prompt,
        "parameters": {"return_full_text": False}
    }

    response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code == 200:
        generated_text = response.json()[0]['generated_text']
        return generated_text.strip()
    else:
        raise HTTPException(status_code=500, detail=f"HuggingFace API Error: {response.text}")

# Retrieval + RAG Answer Generation
def retrieve_and_answer(questions):
    answers = []
    for q in questions:
        docs = vector_store.similarity_search(q, k=5)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are a policy clause extraction assistant. Given the context, extract the exact clause(s) or sentences that answer the user's question. Do not summarize or infer. If the answer is not found in the context, reply 'Not Found'.

Context:
{context}

Question: {q}

Answer:
"""
        answer = call_huggingface_inference(prompt)
        answers.append(answer)

    return answers

@app.post("/api/v1/hackrx/run")
def run_submission(req: RunRequest):
    pdf_path = "temp_doc.pdf"
    download_pdf(req.documents, pdf_path)
    process_document(pdf_path)
    answers = retrieve_and_answer(req.questions)
    return {"answers": answers}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
