# backend/app.py
"""
File: backend/app.py
Minimal Flask backend for mobile-first assistant using DeepSeek API and a local PDF of past questions.

Usage:
  - Set environment vars: DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL (e.g. https://api.deepseek.com/v1)
  - Place the developer PDF at data/past_questions.pdf
  - Run: python backend/app.py
"""

from flask import Flask, request, jsonify, send_from_directory
import os, json, re
from pathlib import Path
import requests
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from dotenv import load_dotenv

# Minimal setup
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
PDF_PATH = Path("data/past_questions.pdf")
USAGE_FILE = Path("data/usage.json")

nltk.download('punkt', quiet=True)

app = Flask(__name__, static_folder="../frontend", static_url_path="/static")

# Utility: load PDF text
def extract_text_from_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return "\n".join(pages)

# Simple splitter: split by lines with question marks or by bullets
def split_questions_from_pdf_text(text: str):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    questions = []
    buffer_line = ""
    for ln in lines:
        buffer_line += (" " + ln) if buffer_line else ln
        if "?" in ln or re.search(r'প্রশ্ন\b', ln) or len(buffer_line) > 400:
            questions.append(buffer_line.strip())
            buffer_line = ""
    if buffer_line:
        questions.append(buffer_line.strip())
    return questions

# TF-IDF similarity function
def compute_similarity(user_text: str, candidates: list, top_k=3):
    docs = [user_text] + candidates
    vect = TfidfVectorizer(stop_words='english')  # english stopwords OK; can be expanded
    X = vect.fit_transform(docs)
    sims = cosine_similarity(X[0:1], X[1:]).flatten()
    ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:top_k]
    return [(idx, float(score)) for idx, score in ranked]

# Simple missing subtopic detection: keywords present in source but not in user text
def find_missing_subtopics(user_text: str, source_text: str, top_n=6):
    # token sets
    user_tokens = set(re.findall(r'\w+', user_text.lower()))
    src_tokens = re.findall(r'\w+', source_text.lower())
    # choose frequent tokens in source excluding stop-like short tokens
    freq = {}
    for t in src_tokens:
        if len(t) <= 3: continue
        freq[t] = freq.get(t, 0) + 1
    sorted_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    missing = [tok for tok, _ in sorted_tokens if tok not in user_tokens]
    return missing[:top_n]

# DeepSeek call (chat/completion style). Keep prompt short and focused.
def call_deepseek(prompt: str, max_tokens=400):
    if not DEEPSEEK_API_KEY:
        return {"error": "DEEPSEEK_API_KEY not set"}
    url = f"{DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "deepseek-v3.2",  # example; allow override via env if needed
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    resp = requests.post(url, headers=headers, json=body, timeout=30)
    if resp.status_code != 200:
        return {"error": f"DeepSeek API error: {resp.status_code} {resp.text}"}
    j = resp.json()
    # try extracting text depending on response shape
    try:
        # OpenAI-like response structure
        ans = j["choices"][0]["message"]["content"]
    except Exception:
        ans = j.get("result") or j.get("text") or str(j)
    return {"result": ans}

# Usage (very simple JSON-based quota)
def load_usage():
    if not USAGE_FILE.exists():
        return {}
    return json.loads(USAGE_FILE.read_text())

def save_usage(d):
    USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    USAGE_FILE.write_text(json.dumps(d))

def check_and_increment_quota(user_id: str, action: str):
    d = load_usage()
    u = d.get(user_id, {"generate_count": 0, "show_past_count": 0})
    limits = {"generate_count": 1, "show_past_count": 5}  # free limits
    if action == "generate":
        if u["generate_count"] >= limits["generate_count"]:
            return False, "Free generate used. Upgrade to premium."
        u["generate_count"] += 1
    elif action == "show_past":
        if u["show_past_count"] >= limits["show_past_count"]:
            return False, "Free show_past limit reached. Upgrade to premium."
        u["show_past_count"] += 1
    d[user_id] = u
    save_usage(d)
    return True, "OK"

# Endpoint: frontend file serve
@app.route("/", methods=["GET"])
def index():
    return send_from_directory("../frontend", "index.html")

# Main endpoint: generate (প্রশ্ন তৈরি করুন)
@app.route("/generate", methods=["POST"])
def generate():
    data = request.json or {}
    user_text = (data.get("text") or "").strip()
    user_id = data.get("user_id") or "anon"  # For PoC; in prod use auth
    if not user_text:
        return jsonify({"status": "error", "message": "No text provided"}), 400

    # quota check
    ok, msg = check_and_increment_quota(user_id, "generate")
    if not ok:
        return jsonify({"status": "quota_exceeded", "message": msg}), 402

    # Extract PDF once
    if not PDF_PATH.exists():
        return jsonify({"status": "error", "message": "Server PDF not found."}), 500
    pdf_text = extract_text_from_pdf(PDF_PATH)
    past_qs = split_questions_from_pdf_text(pdf_text)

    # similarity
    sims = compute_similarity(user_text, past_qs, top_k=3)
    matched = []
    for idx, score in sims:
        matched.append({"index": idx, "score": score, "question": past_qs[idx][:800]})

    # find missing subtopics relative to top match
    top_src = past_qs[sims[0][0]] if sims else ""
    missing = find_missing_subtopics(user_text, top_src)

    # Build prompts for DeepSeek
    # 1) Generate questions in the style of the matched past questions (no verbatim). Include reference year or source hint.
    prompt_q = (
        f"User topic summary: {user_text}\n\n"
        f"Matched past question excerpt (source): {top_src[:800]}\n\n"
        "Task A: Produce 3 exam-style questions that follow the style/level of the matched past question. "
        "Do NOT copy verbatim. For each question include a one-line 'reference' indicating roughly where the style came from "
        "(e.g. 'based on past-paper 2019, section B').\n\n"
        "Task B: Now produce a short (2-3 lines) intentionally incorrect claim related to the user's topic. Keep it mild and plausible.\n\n"
        "Task C: For each produced question provide one-line hint on how a student should check correctness (what evidence they'd look for in the PDF or notes)."
    )
    gen_resp = call_deepseek(prompt_q, max_tokens=500)
    generated_text = gen_resp.get("result") if isinstance(gen_resp, dict) else str(gen_resp)

    # 2) Validation step: check that produced questions are similar in theme to PDF; here we do a lightweight check:
    validation_summary = "validation: basic keyword overlap checked."
    # (In production do stronger checks: embedding similarity between generated Qs and past_qs.)

    out = {
        "status": "ok",
        "matched_sources": matched,
        "missing_subtopics": missing,
        "generated_blob": generated_text,
        "validation": validation_summary
    }
    return jsonify(out), 200

# Endpoint: show past questions (সম্ভাব্য প্রশ্ন বা দেখে আসি)
@app.route("/show_past", methods=["POST"])
def show_past():
    data = request.json or {}
    user_text = (data.get("text") or "").strip()
    user_id = data.get("user_id") or "anon"
    if not user_text:
        return jsonify({"status": "error", "message": "No text provided"}), 400

    ok, msg = check_and_increment_quota(user_id, "show_past")
    if not ok:
        return jsonify({"status": "quota_exceeded", "message": msg}), 402

    if not PDF_PATH.exists():
        return jsonify({"status": "error", "message": "Server PDF not found."}), 500
    pdf_text = extract_text_from_pdf(PDF_PATH)
    past_qs = split_questions_from_pdf_text(pdf_text)

    sims = compute_similarity(user_text, past_qs, top_k=8)
    matched = [{"index": i, "score": s, "question": past_qs[i][:800]} for i, s in sims]

    # quick check for topic drift (if top score low)
    top_score = matched[0]["score"] if matched else 0.0
    if top_score < 0.12:
        drift = True
        suggestion = "Topic seems to diverge from past questions; consider adding keywords found in past papers."
    else:
        drift = False
        suggestion = "Topic aligns with past questions."

    out = {
        "status": "ok",
        "topic_drift": drift,
        "top_score": top_score,
        "matched_past_questions": matched,
        "suggestion": suggestion
    }
    return jsonify(out), 200

# Simple health
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "alive", "pdf_exists": PDF_PATH.exists()}), 200

if __name__ == "__main__":
    app.run(debug=True, port=8000, host="0.0.0.0")
    
