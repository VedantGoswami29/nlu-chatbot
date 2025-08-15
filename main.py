from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict
import os
import re
import torch

# Load environment variables
load_dotenv()

app = FastAPI(
    title="NLU Chatbot Server",
    description="A FastAPI server using Sentence Transformers for FAQ matching.",
    version="1.0.0",
)

# --- CORS ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBALS ---
faq_data: List[Dict] = []
faq_embeddings = None
model = None

# --- PARSE .TXT FAQ FILES ---
def parse_txt_data(text_data: str) -> List[Dict]:
    faqs = []
    records = re.split(r'(^\s*Question ID:)', text_data, flags=re.IGNORECASE | re.MULTILINE)
    for i in range(1, len(records), 2):
        record_content = (records[i] + records[i+1]).strip()
        record_content = record_content.split('________________')[0].strip()
        if not record_content:
            continue

        faq_entry = {}
        current_key = None

        for line in record_content.split('\n'):
            match = re.match(r'^\s*([\w\s/().-]+):\s*(.*)', line)
            if match:
                key = match.group(1).strip().lower().replace(" ", "_").replace("-", "_")
                value = match.group(2).strip()

                if key == 'question_id':
                    faq_entry['question_id'] = value
                    current_key = None
                    continue

                faq_entry[key] = value
                current_key = key
            elif current_key and line.strip() and current_key in faq_entry:
                faq_entry[current_key] += " " + line.strip()

        if 'keyword' in faq_entry:
            faq_entry['keywords'] = [kw.strip() for kw in faq_entry['keyword'].split(',')]
            del faq_entry['keyword']
        elif 'keywords' in faq_entry:
            faq_entry['keywords'] = [kw.strip() for kw in faq_entry['keywords'].split(',')]

        if 'question' in faq_entry and 'answer' in faq_entry:
            faqs.append(faq_entry)
    return faqs

# --- STARTUP ---
@app.on_event("startup")
async def startup_event():
    global faq_data, faq_embeddings, model

    # Get config from .env
    faq_directory = os.environ.get("FAQ_PATH", "faq_data")
    sentence_model_name = os.environ.get("SENTENCE_MODEL", "all-MiniLM-L6-v2")

    # Optional: custom cache location
    cache_dir = os.environ.get("SENTENCE_TRANSFORMERS_HOME")
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    if not os.path.isdir(faq_directory):
        print(f"--- WARNING: Directory '{faq_directory}' not found. ---")
        return

    # Load all FAQ text files
    all_text_data = ""
    print(f"--- Loading data from '{faq_directory}'... ---")
    for filename in sorted(os.listdir(faq_directory)):
        if filename.endswith(".txt"):
            try:
                with open(os.path.join(faq_directory, filename), "r", encoding="utf-8") as f:
                    all_text_data += f.read() + "\n\n"
            except Exception as e:
                print(f"--- ERROR reading file {filename}: {e} ---")

    if not all_text_data.strip():
        print("--- WARNING: No data found in FAQ files. ---")
        return

    # Parse FAQs
    faq_data = parse_txt_data(all_text_data)
    if not faq_data:
        print("--- WARNING: No valid FAQ entries parsed. ---")
        return

    # Load model
    print(f"--- Loading Sentence Transformer model: {sentence_model_name} ---")
    model = SentenceTransformer(sentence_model_name)

    # Encode questions
    questions = [item['question'] for item in faq_data]
    faq_embeddings = model.encode(questions, convert_to_tensor=True)

    print(f"--- Successfully loaded {len(faq_data)} Q&A pairs. ---")

# --- ROOT ---
@app.get("/")
def read_root():
    return {"status": "Chatbot server is running", "faqs_loaded": len(faq_data)}

# --- ASK ---
@app.post("/ask/")
async def ask_question(query: str = Body(..., embed=True)):
    global faq_data, faq_embeddings, model

    if not faq_data or faq_embeddings is None:
        raise HTTPException(status_code=503, detail="Server not ready.")

    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, faq_embeddings)[0]
    best_match_index = int(torch.argmax(cos_scores))
    confidence = float(cos_scores[best_match_index])

    SIMILARITY_THRESHOLD = 0.50
    if confidence > SIMILARITY_THRESHOLD:
        return {
            "answer": faq_data[best_match_index]['answer'],
            "matched_question": faq_data[best_match_index]['question'],
            "confidence": confidence
        }
    else:
        return {
            "answer": "I am unable to answer that. Please contact us on: 7574949494 or 9099951160.",
            "matched_question": None,
            "confidence": confidence
        }