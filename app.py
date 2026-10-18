"""
KOISS Chatbot -- Flask API wrapper
Exposes the same retrieval + Groq logic from engage_chatbot.py as a
web endpoint, so a WordPress chat widget can call it.

Setup:
    pip install flask flask-cors sentence-transformers requests
    export GROQ_API_KEY="your-key-here"

Run locally:
    python3 app.py
    -> runs at http://127.0.0.1:5000

Test it:
    curl -X POST http://127.0.0.1:5000/chat -H "Content-Type: application/json" -d '{"message": "hi"}'
"""

import os
import numpy as np
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_BASE = "https://api.groq.com/openai/v1"
GROQ_URL = f"{GROQ_BASE}/chat/completions"
PREFERRED_MODELS = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "llama-3.1-8b-instant", "openai/gpt-oss-20b"]

if not GROQ_API_KEY:
    raise SystemExit("GROQ_API_KEY not found. Set it as an environment variable first.")


def pick_available_model():
    resp = requests.get(f"{GROQ_BASE}/models", headers={"Authorization": f"Bearer {GROQ_API_KEY}"}, timeout=15)
    resp.raise_for_status()
    available = {m["id"] for m in resp.json()["data"]}
    for candidate in PREFERRED_MODELS:
        if candidate in available:
            return candidate
    return sorted(available)[0]


GROQ_MODEL = pick_available_model()
print(f"Using model: {GROQ_MODEL}")

knowledge_base = [
    {"topic": "Admissions", "q": "How do I apply for a course?",
     "a": "Applications are submitted through the online student portal. You'll need your academic transcripts and a copy of your ID."},
    {"topic": "Fees", "q": "When are tuition fees due?",
     "a": "Tuition fees are due at the start of each trimester, usually within the first two weeks of enrolment."},
    {"topic": "Campus", "q": "Where is the student lounge?",
     "a": "The student lounge is on Level 2 of the main building, next to the library entrance."},
    {"topic": "Courses", "q": "Can I switch majors after enrolling?",
     "a": "Yes, major changes can be requested through student services before the census date of a trimester."},
    {"topic": "IT Support", "q": "How do I reset my student email password?",
     "a": "Go to the IT self-service portal and select 'Reset Password'. You'll need your student ID to verify identity."},
    {"topic": "Events", "q": "Are there any events happening this month?",
     "a": "Orientation week runs in the first week of the trimester, with a welcome BBQ, campus tour, and club sign-ups."},
]

print("Loading retrieval model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
corpus = [item["q"] + " " + item["a"] for item in knowledge_base]
kb_vectors = model.encode(corpus, normalize_embeddings=True)
RETRIEVAL_THRESHOLD = 0.45

SYSTEM_PROMPT = """You are KOISS Bot, a warm and friendly assistant for KOI students.
Speak casually and briefly, like a helpful older student, not a formal FAQ page.
Keep answers short, 2-3 sentences max, unless asked for more detail.
If given verified context, base your answer strictly on it -- do not add
facts that aren't in the context. If no verified context is given and the
question is factual (dates, requirements, procedures, policies), do NOT
guess or invent specifics -- say you're not certain and suggest contacting
student services. For greetings and casual chat, just respond naturally."""


def retrieve(user_message):
    query_vec = model.encode([user_message], normalize_embeddings=True)[0]
    scores = np.dot(kb_vectors, query_vec)
    best_idx = int(np.argmax(scores))
    return knowledge_base[best_idx], float(scores[best_idx])


def ask_groq(user_message, context=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context:
        messages.append({"role": "system", "content": f"Verified context for this question: {context}"})
    messages.append({"role": "user", "content": user_message})
    resp = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={"model": GROQ_MODEL, "messages": messages, "temperature": 0.7, "max_tokens": 200},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def chatbot(user_message):
    item, score = retrieve(user_message)
    if score >= RETRIEVAL_THRESHOLD:
        context = f"[{item['topic']}] {item['a']}"
        return ask_groq(user_message, context=context), score
    return ask_groq(user_message), score


app = Flask(__name__)
CORS(app)  # allows WordPress (a different domain) to call this API


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_message = (data or {}).get("message", "").strip()
    if not user_message:
        return jsonify({"error": "message is required"}), 400
    reply, score = chatbot(user_message)
    return jsonify({"reply": reply, "confidence": round(score, 2)})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": GROQ_MODEL})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
