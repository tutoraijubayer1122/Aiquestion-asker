from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import requests
import re
import logging

# -------------------------------------------------
# App + CORS + Logging
# -------------------------------------------------
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# DeepSeek API Config
# -------------------------------------------------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"


# -------------------------------------------------
# Helper Functions
# -------------------------------------------------
def sanitize(text):
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def word_count(text):
    return len(text.split())


def extract_json(text):
    """
    Extract JSON from LLM output.
    """
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None
        cleaned = text[start:end+1]
        return json.loads(cleaned)
    except:
        return None


# -------------------------------------------------
# Health Route
# -------------------------------------------------
@app.route("/", methods=["GET"])
def health():
    return "DeepSeek backend live", 200


# -------------------------------------------------
# Main API Route
# -------------------------------------------------
@app.route("/generate", methods=["POST"])
def generate():
    logger.info("---- /generate HIT ----")
    logger.info(f"Incoming JSON: {request.get_json()}")

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Invalid request — missing 'text'"}), 400

    student_text = sanitize(data["text"])

    if word_count(student_text) > 40:
        return jsonify({"error": "Maximum 40 words allowed"}), 400

    # -----------------------------
    # Build LLM Prompt
    # -----------------------------
    prompt = f"""
You must output STRICT JSON.

Student summary:
{student_text}

Output 3 exam-style questions in this JSON schema:

{{
  "analysis": {{
    "missing": ["..."],
    "unnecessary": ["..."],
    "core_points": ["..."]
  }},
  "questions": [
    {{
      "q_no": 1,
      "question": "...",
      "hint": "...",
      "guidance": ["...", "..."],
      "source_pointer": "..."
    }},
    {{
      "q_no": 2,
      "question": "...",
      "hint": "...",
      "guidance": ["...", "..."],
      "source_pointer": "..."
    }},
    {{
      "q_no": 3,
      "question": "...",
      "hint": "...",
      "guidance": ["...", "..."],
      "source_pointer": "..."
    }}
  ]
}}

Return ONLY pure JSON — no explanations.
"""

    # -----------------------------
    # Call DeepSeek API
    # -----------------------------
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }

    try:
        logger.info("Calling DeepSeek API...")
        r = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=20)
        logger.info(f"DeepSeek HTTP status: {r.status_code}")
        r.raise_for_status()
        result = r.json()
    except Exception as e:
        logger.error(f"DeepSeek API failed: {e}")
        return jsonify({"error": "LLM call failed", "detail": str(e)}), 502

    # Extract model response
    try:
        raw = result["choices"][0]["message"]["content"]
        logger.info(f"Raw LLM output: {raw}")
    except:
        return jsonify({"error": "Unexpected DeepSeek response", "raw": result}), 500

    parsed = extract_json(raw)

    if not parsed:
        return jsonify({
            "error": "JSON parsing failed",
            "raw": raw
        }), 200

    if "questions" not in parsed or len(parsed["questions"]) != 3:
        return jsonify({
            "error": "Bad structure from model — must contain exactly 3 questions",
            "raw": raw
        }), 200

    return jsonify(parsed), 200


# -------------------------------------------------
# Entry Point
# -------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
