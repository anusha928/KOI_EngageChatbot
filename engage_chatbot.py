"""
KOISS Chatbot -- Phase 1 prototype
Combines: sentence-transformer retrieval + Groq LLM for natural responses.

Setup:
    pip install sentence-transformers requests
    export GROQ_API_KEY="your-key-here"      (Mac/Linux)
    setx GROQ_API_KEY "your-key-here"        (Windows, then restart terminal)

Run:
    python3 engage_chatbot.py

NEVER hardcode your API key directly in this file -- always load it from
an environment variable, especially before this ever goes near GitHub or
a shared repo.
"""

import os
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_BASE = "https://api.groq.com/openai/v1"
GROQ_URL = f"{GROQ_BASE}/chat/completions"

# Preference order -- script will use the first one your key actually has access to.
PREFERRED_MODELS = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
    "gemma2-9b-it",
]

if not GROQ_API_KEY:
    raise SystemExit("GROQ_API_KEY not found. Set it as an environment variable first.")


def pick_available_model():
    resp = requests.get(f"{GROQ_BASE}/models", headers={"Authorization": f"Bearer {GROQ_API_KEY}"}, timeout=15)
    resp.raise_for_status()
    available = {m["id"] for m in resp.json()["data"]}
    for candidate in PREFERRED_MODELS:
        if candidate in available:
            return candidate
    # fall back to whatever's first in the account's list
    return sorted(available)[0]


GROQ_MODEL = pick_available_model()
print(f"Using model: {GROQ_MODEL}")

# ---- 1. Knowledge base -- expand this with real content + your survey results ----
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
    {"topic": "Events", "q": "How do I join a student club?",
     "a": "Club sign-ups happen during orientation week and stay open online through the student portal all trimester."},
]

# ---- 2. Retrieval setup ----
print("Loading retrieval model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
corpus = [item["q"] + " " + item["a"] for item in knowledge_base]
kb_vectors = model.encode(corpus, normalize_embeddings=True)

RETRIEVAL_THRESHOLD = 0.4  # below this, treat it as chit-chat, not a real question


def retrieve(user_message):
    query_vec = model.encode([user_message], normalize_embeddings=True)[0]
    scores = np.dot(kb_vectors, query_vec)
    best_idx = int(np.argmax(scores))
    return knowledge_base[best_idx], float(scores[best_idx])


# ---- 3. Personality -- this is where the "natural rhythm" comes from ----
SYSTEM_PROMPT = """You are KOISS Bot, a warm and friendly assistant for KOI students.
Speak casually and briefly, like a helpful older student, not a formal FAQ page.
Use light encouragement and ask a natural follow-up question when it fits
(e.g. "are you new to KOI?", "what are you looking to explore first?").
Keep answers short, 2-3 sentences max, unless the student asks for more detail.
If you're given verified context, base your answer on it -- don't add facts
that aren't in the context. If no context is given, just chat naturally,
and if asked something factual you don't have context for, be honest and
suggest they contact student services."""


def ask_groq(user_message, context=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context:
        messages.append({
            "role": "system",
            "content": f"Verified context for this question: {context}"
        })
    messages.append({"role": "user", "content": user_message})

    response = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={"model": GROQ_MODEL, "messages": messages, "temperature": 0.7, "max_tokens": 200},
        timeout=15,
    )
    if response.status_code != 200:
        print(f"\n[Groq API error {response.status_code}]: {response.text}\n")
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def chatbot(user_message):
    item, score = retrieve(user_message)
    if score >= RETRIEVAL_THRESHOLD:
        context = f"[{item['topic']}] {item['a']}"
        return ask_groq(user_message, context=context), score
    else:
        return ask_groq(user_message), score


# ---- 4. Interactive chat loop ----
if __name__ == "__main__":
    print("\n=== KOISS Chatbot (Phase 1 prototype) ===")
    print("Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        reply, score = chatbot(user_input)
        print(f"Bot: {reply}  [match confidence: {score:.2f}]\n")
