
#!/usr/bin/env python3
"""
Flask backend for Practice Question Generator
- POST /generate  { "text": "<student summary up to 40 words>" }
- returns JSON with analysis + exactly 3 questions (or helpful errors)
"""

import os
import json
import re
import logging
import difflib
from typing import Optional, Dict, Any, List
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

# ---------------------------
# Configuration & Logging
# ---------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ai_question_backend")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # must be set in Render environment
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
# Default to an accessible model; change by setting MODEL env var.
MODEL = os.getenv("MODEL", "gpt-3.5-turbo")

# timeout for LLM HTTP call (seconds)
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "20"))

# ---------------------------
# Minimal "previous patterns"
# You can extend this list or read from a JSON file in production.
# ---------------------------
PREVIOUS_PATTERNS = [
    "Explain the main causes of X. Give two examples and evaluate the consequences.",
    "Define the concept and provide a diagram. Then solve a short problem applying the concept.",
    "Compare and contrast A and B; include advantages and disadvantages and one real-world example.",
]

# ---------------------------
# Flask app
# ---------------------------
app = Flask(__name__)
CORS(app)  # allow requests from any origin (frontend hosted elsewhere)

# ---------------------------
# Helpers
# ---------------------------
def sanitize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def words_count(s: str) -> int:
    if not s:
        return 0
    return len(s.split())

def build_prompt(user_text: str, previous_patterns: List[str]) -> str:
    """
    Build the instruction prompt for the LLM. Keep format stable so parsing is easier.
    """
    prompt = f"""
You are an expert exam-writer and pedagogue. A student has supplied this text (they learned it but did not ask specific questions):
---
{user_text}
---

Task:
1) Identify missing, unnecessary, and core/main points in the student's supplied explanation. Produce a short bullet list:
   - missing: (concise)
   - unnecessary: (concise)
   - core points: (concise)

2) Using the student's supplied material and the "previous question patterns" below, produce exactly THREE (3) practice questions that emulate the style and difficulty of those previous patterns (e.g., board/exam/admission test style). Each question must:
   - be clearly numbered (1/2/3),
   - be answerable in 10-30 lines (~150-300 words),
   - be focused on the student's topic,
   - include one brief hint line (2-3 words) following the question.

3) For each question, provide:
   - a short guidance/outline for the answer (3–6 bullet points with the main points the student should include),
   - one pointer to where the student could find the answer (e.g., "Use concept X, see definition Y, apply example Z").

Previous question patterns (use these to shape phrasing and structure):
{json.dumps(previous_patterns, ensure_ascii=False, indent=2)}

Return output STRICTLY as JSON with this schema:
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
      "guidance": ["...","..."],
      "source_pointer": "..."
    }},
    {{
      "q_no": 2,
      "question": "...",
      "hint": "...",
      "guidance": ["...","..."],
      "source_pointer": "..."
    }},
    {{
      "q_no": 3,
      "question": "...",
      "hint": "...",
      "guidance": ["...","..."],
      "source_pointer": "..."
    }}
  ]
}}
Be strict JSON only. No extra commentary.
"""
    return prompt

def call_openai_chat(prompt: str) -> str:
    """
    Call OpenAI Chat Completions endpoint (or compatible). Returns assistant text.
    Raises requests.exceptions.RequestException or ValueError for caller to handle.
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set in environment")

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that follows JSON output strictly."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 900,
        "n": 1,
    }

    logger.debug("Calling OpenAI: model=%s url=%s", MODEL, OPENAI_API_URL)
    resp = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=LLM_TIMEOUT)
    # Raise for HTTP errors (4xx/5xx)
    resp.raise_for_status()
    data = resp.json()
    # Navigate typical OpenAI response shape
    if "choices" in data and len(data["choices"]) > 0:
        # assistant message text
        # For models that return 'message' objects:
        choice = data["choices"][0]
        if "message" in choice and "content" in choice["message"]:
            return choice["message"]["content"]
        # For models that use 'text':
        if "text" in choice:
            return choice["text"]
    raise ValueError("Unexpected response shape from LLM")

def minimal_json_fix(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    Try to extract JSON from raw_text. Return parsed dict or None.
    """
    if not raw_text or not isinstance(raw_text, str):
        return None
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1:
        return None
    candidate = raw_text[start:end+1]
    # Basic cleanups
    candidate_clean = re.sub(r",\s*}", "}", candidate)
    candidate_clean = re.sub(r",\s*]", "]", candidate_clean)
    try:
        return json.loads(candidate_clean)
    except Exception as ex:
        logger.debug("minimal_json_fix parse failed: %s", ex)
        return None

def similarity_to_patterns(text: str, patterns: List[str]) -> float:
    ratios = []
    for p in patterns:
        try:
            r = difflib.SequenceMatcher(None, text.lower(), p.lower()).ratio()
            ratios.append(r)
        except Exception:
            continue
    return max(ratios) if ratios else 0.0

# ---------------------------
# Routes
# ---------------------------
@app.route("/", methods=["GET"])
def health():
    return "Practice Question Generator backend is live", 200

@app.route("/generate", methods=["POST"])
def generate():
    """
    Main endpoint:
    Request JSON: { "text": "<student summary up to 40 words>" }
    Response: application/json (analysis + questions)
    """
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON request"}), 400

    user_text = sanitize_text(payload.get("text", "")) if isinstance(payload, dict) else ""
    logger.info("Received /generate request, length=%d", len(user_text))

    if not user_text:
        return jsonify({"error": "No text provided."}), 400

    if words_count(user_text) > 40:
        return jsonify({"error": "Input exceeds 40 words limit."}), 400

    # Build prompt
    prompt = build_prompt(user_text, PREVIOUS_PATTERNS)

    # Call LLM
    try:
        raw_response = call_openai_chat(prompt)
    except requests.exceptions.HTTPError as e:
        # OpenAI returned non-200
        logger.error("LLM HTTPError: %s", e)
        try:
            detail = e.response.json()
        except Exception:
            detail = str(e)
        return jsonify({"error": "LLM call failed", "detail": detail}), 502
    except requests.exceptions.RequestException as e:
        logger.error("LLM request exception: %s", e)
        return jsonify({"error": "LLM call failed", "detail": str(e)}), 502
    except ValueError as e:
        logger.error("LLM config/value error: %s", e)
        return jsonify({"error": "LLM call failed", "detail": str(e)}), 500
    except Exception as e:
        logger.exception("Unexpected error calling LLM")
        return jsonify({"error": "LLM call failed", "detail": str(e)}), 500

    # Attempt parse
    parsed = minimal_json_fix(raw_response)
    if parsed is None:
        # Provide raw text but mark warning
        return jsonify({
            "warning": "Could not parse JSON from model output. Returning raw text.",
            "raw": raw_response
        }), 200

    # Validate three questions
    qs = parsed.get("questions", [])
    if not isinstance(qs, list) or len(qs) != 3:
        # Try to coerce/refine by asking LLM to fix
        refine_prompt = f"""
The previous output contained {len(qs) if isinstance(qs, list) else 'unknown'} questions. The user needs EXACTLY 3 questions.
Take this previous JSON output and produce corrected JSON matching the Output format exactly,
ensuring 'questions' is a list of exactly 3 question objects (q_no 1..3). Keep 'analysis' intact if present.
Return JSON only.
Previous JSON string: {json.dumps(parsed, ensure_ascii=False)}
"""
        try:
            refine_raw = call_openai_chat(refine_prompt)
            refined = minimal_json_fix(refine_raw)
            if refined is not None and isinstance(refined.get("questions", []), list) and len(refined["questions"]) == 3:
                parsed = refined
                qs = parsed.get("questions", [])
        except Exception as e:
            logger.warning("Refine attempt failed: %s", e)

    # Similarity scoring
    similarities = []
    for q in qs:
        qtext = q.get("question", "") if isinstance(q, dict) else ""
        similarities.append(similarity_to_patterns(qtext, PREVIOUS_PATTERNS))

    parsed["_meta"] = {
        "similarity_scores": similarities,
        "max_similarity": max(similarities) if similarities else 0.0
    }

    # Final response
    return jsonify(parsed), 200

# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting app on 0.0.0.0:%s model=%s", port, MODEL)
    app.run(host="0.0.0.0", port=port, debug=False)
