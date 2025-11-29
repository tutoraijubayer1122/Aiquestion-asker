# app.py
from flask import Flask, request, jsonify
import os
import json
import difflib
import re
import requests

app = Flask(__name__)

# ----- Configuration -----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Set this in your environment (or Colab cell)
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"  # or the endpoint you use
MODEL = "gpt-4o"  # replace with your available model; keep configurable

# Sample stored previous question patterns (minimal examples).
# In production, store a larger JSON file and update via admin.
PREVIOUS_PATTERNS = [
    "Explain the main causes of X. Give two examples and evaluate the consequences.",
    "Define the concept and provide a diagram. Then solve a short problem applying the concept.",
    "Compare and contrast A and B; include advantages and disadvantages and one real-world example.",
]

# ---------- Helpers ----------
def sanitize_text(s):
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def words_count(s):
    return len(s.split())

def build_prompt(user_text, previous_patterns):
    """
    Build the LLM prompt for generating three exam-style questions + guidance.
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

2) Using the student's supplied material and the "previous question patterns" below, produce exactly THREE (3) practice questions that emulate the style and difficulty of those previous patterns (e.g., board/exam/admission test style). Each question should:
   - be clearly numbered (1/2/3),
   - be answerable in 10-30 lines (or ~150-300 words) in an exam,
   - be focused on the student's topic,
   - include one brief hint line (2-3 words) following the question.

3) For each question, provide:
   - a short guidance/outline for the answer (3–6 bullet points with the main points the student should include),
   - one pointer to where the student could find the answer (match format of 'previous question patterns' - e.g., "Use concept X, see definition Y, apply example Z").

Prioritize clarity, exam-style phrasing, and alignment with the patterns provided.

Previous question patterns (use these to shape phrasing and structure):
{json.dumps(previous_patterns, ensure_ascii=False, indent=2)}

Output format (JSON only, no extra commentary):
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
    ...
  ]
}}
"""
    return prompt

def call_openai_chat(prompt):
    """
    Call to OpenAI-style chat completion. Replace with your provider if different.
    """
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that follows JSON output strictly."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 900,
        "n": 1,
    }
    resp = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    # This reads the assistant message (assumes OpenAI chat response shape).
    return data["choices"][0]["message"]["content"]

def minimal_json_fix(raw_text):
    """
    Attempt to extract JSON block from the model output and parse it.
    If parsing fails, try to find the first { ... } and parse.
    """
    # Try to locate a JSON block
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1:
        candidate = raw_text[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            # Remove trailing commas and stray newlines
            candidate_clean = re.sub(r",\s*}", "}", candidate)
            candidate_clean = re.sub(r",\s*]", "]", candidate_clean)
            try:
                return json.loads(candidate_clean)
            except Exception:
                pass
    # As fallback, attempt to return None
    return None

def similarity_to_patterns(text, patterns):
    """
    Simple similarity: compare the question strings to previous patterns using difflib.
    Return the max ratio (0.0 - 1.0).
    """
    ratios = []
    for p in patterns:
        r = difflib.SequenceMatcher(None, text.lower(), p.lower()).ratio()
        ratios.append(r)
    return max(ratios) if ratios else 0.0

# ---------- Route ----------
@app.route("/generate", methods=["POST"])
def generate():
    data = request.json or {}
    user_text = data.get("text", "")
    user_text = sanitize_text(user_text)

    if not user_text:
        return jsonify({"error": "No text provided."}), 400

    if words_count(user_text) > 40:
        return jsonify({"error": "Input exceeds 40 words limit."}), 400

    # Build prompt
    prompt = build_prompt(user_text, PREVIOUS_PATTERNS)

    # Call the LLM
    try:
        raw_response = call_openai_chat(prompt)
    except Exception as e:
        return jsonify({"error": "LLM call failed", "detail": str(e)}), 500

    parsed = minimal_json_fix(raw_response)
    if parsed is None:
        # If parsing failed, return raw_response as fallback (but inform)
        return jsonify({
            "warning": "Could not parse JSON from model output. Returning raw text.",
            "raw": raw_response
        }), 200

    # Validate we got exactly 3 questions
    qs = parsed.get("questions", [])
    if len(qs) != 3:
        # Try to enforce by regenerating with stricter instruction — simple refinement:
        # Build small refine prompt that asks to keep exactly 3 questions.
        refine_prompt = f"""
The previous output contained {len(qs)} questions. The user needs EXACTLY 3 questions.
Take this previous JSON output and produce corrected JSON matching the Output format exactly,
ensuring 'questions' is a list of exactly 3 question objects (1..3). Keep the 'analysis' section intact,
adjust questions only as needed to make three. Return JSON only.
Previous JSON (as string): {json.dumps(parsed, ensure_ascii=False)}
"""
        try:
            refine_resp = call_openai_chat(refine_prompt)
            refined_parsed = minimal_json_fix(refine_resp)
            if refined_parsed:
                parsed = refined_parsed
                qs = parsed.get("questions", [])
        except Exception:
            pass

    # Now similarity check: ensure at least one question is similar to previous patterns.
    # If not, we mark 'may_need_refinement' and return similarity scores so frontend can display.
    similarities = []
    for q in qs:
        qtext = q.get("question", "")
        similarities.append(similarity_to_patterns(qtext, PREVIOUS_PATTERNS))

    parsed["_meta"] = {
        "similarity_scores": similarities,
        "max_similarity": max(similarities) if similarities else 0.0
    }

    return jsonify(parsed), 200

if __name__ == "__main__":
    # For local debug (not production)
    app.run(host="0.0.0.0", port=5000, debug=True)
