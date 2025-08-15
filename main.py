from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import os
import re
import torch

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- GLOBALS ---
faq_data = []
faq_embeddings = None
model = None

# --- PARSE .TXT FAQ FILES ---
def parse_txt_data(text_data: str):
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

# --- LOAD DATA & MODEL ---
def load_faq_and_model():
    global faq_data, faq_embeddings, model

    faq_directory = os.environ.get("FAQ_PATH", "faq_data")
    sentence_model_name = os.environ.get("SENTENCE_MODEL", "all-MiniLM-L6-v2")

    cache_dir = os.environ.get("SENTENCE_TRANSFORMERS_HOME")
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    if not os.path.isdir(faq_directory):
        print(f"--- WARNING: Directory '{faq_directory}' not found. ---")
        return

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

    faq_data = parse_txt_data(all_text_data)
    if not faq_data:
        print("--- WARNING: No valid FAQ entries parsed. ---")
        return

    print(f"--- Loading Sentence Transformer model: {sentence_model_name} ---")
    model = SentenceTransformer(sentence_model_name)

    questions = [item['question'] for item in faq_data]
    faq_embeddings = model.encode(questions, convert_to_tensor=True)

    print(f"--- Successfully loaded {len(faq_data)} Q&A pairs. ---")

# --- ROUTES ---
@app.route("/", methods=["GET"])
def root():
    return render_template('index.html')

@app.route("/ask/", methods=["POST"])
def ask_question():
    global faq_data, faq_embeddings, model

    if not faq_data or faq_embeddings is None:
        return jsonify({"error": "Server not ready."}), 503

    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' field."}), 400

    query = data["query"].strip()
    if not query:
        return jsonify({"error": "Query cannot be empty."}), 400

    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, faq_embeddings)[0]
    best_match_index = int(torch.argmax(cos_scores))
    confidence = float(cos_scores[best_match_index])

    SIMILARITY_THRESHOLD = 0.50
    if confidence > SIMILARITY_THRESHOLD:
        return jsonify({
            "answer": faq_data[best_match_index]['answer'],
            "matched_question": faq_data[best_match_index]['question'],
            "confidence": confidence
        })
    else:
        return jsonify({
            "answer": "I am unable to answer that. Please contact us on: 7574949494 or 9099951160.",
            "matched_question": None,
            "confidence": confidence
        })

if __name__ == "__main__":
    load_faq_and_model()
    app.run(host="0.0.0.0", port=8000, debug=False)