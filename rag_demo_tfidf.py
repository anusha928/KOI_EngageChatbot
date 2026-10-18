"""
Minimal RAG-style chatbot demo (v1) -- near-zero-cost approach using TF-IDF.

Pipeline:
1. Knowledge base (dummy FAQ data -- replace with real content later)
2. Retrieval: TF-IDF + cosine similarity (no API cost, runs fully offline)
3. Generation: normally you'd send the retrieved chunk + question to a
   free-tier LLM API (Groq / Gemini). generate() below is a placeholder
   that just returns the retrieved chunk directly.

See rag_demo_v2.py for the embeddings-based upgrade and why it matters.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
]

corpus = [item["q"] + " " + item["a"] for item in knowledge_base]
vectorizer = TfidfVectorizer(stop_words="english")
kb_vectors = vectorizer.fit_transform(corpus)


def retrieve(user_question, top_k=1):
    query_vec = vectorizer.transform([user_question])
    scores = cosine_similarity(query_vec, kb_vectors)[0]
    best_idx = scores.argsort()[::-1][:top_k]
    return [(knowledge_base[i], scores[i]) for i in best_idx]


def generate(user_question, retrieved_item, score):
    if score < 0.1:
        return "I'm not confident I know this one -- try rephrasing, or contact student services directly."
    return f"[{retrieved_item['topic']}] {retrieved_item['a']}"


def chatbot(user_question):
    best_item, score = retrieve(user_question)[0]
    answer = generate(user_question, best_item, score)
    return answer, score


demo_questions = [
    "how do i submit my application",
    "when do i need to pay fees",
    "is there a place to hang out on campus",
    "can I change my major",
    "forgot my email password, what do i do",
    "what's the weather like today",  # deliberately off-topic
]

if __name__ == "__main__":
    print("=== RAG Chatbot Demo (TF-IDF) ===\n")
    for q in demo_questions:
        answer, score = chatbot(q)
        print(f"Q: {q}")
        print(f"A: {answer}  (confidence: {score:.2f})\n")
