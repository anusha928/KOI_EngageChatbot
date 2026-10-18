"""
RAG chatbot demo v2 -- real embeddings instead of TF-IDF.

Pipeline:
1. Knowledge base (same dummy FAQ data as before)
2. Retrieval: GloVe word embeddings (via gensim) averaged into a sentence
   vector, + cosine similarity. Free, runs locally, downloaded once.
3. Generation: still a placeholder -- swap for a free-tier LLM API call
   (Groq / Gemini) later. Retrieval logic below doesn't change when you do.

Note: averaged word embeddings are a step up from TF-IDF but still not as
strong as a real sentence-transformer model (which understands word order
and context, not just averaged word meaning). Good enough to demonstrate
the concept without a heavy download.
"""

import re
import numpy as np
import gensim.downloader as api
from sklearn.feature_extraction.text import TfidfVectorizer

# ---- 1. Knowledge base (same as before) ----
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

# ---- 2. Load embedding model (cached locally after first download) ----
print("Loading embedding model...")
word_vectors = api.load("glove-wiki-gigaword-50")


# Fit TF-IDF just to get sensible word weights (rare words = more weight,
# common words like "the"/"student" = less weight). We're not using TF-IDF
# for matching this time -- just borrowing its weights.
corpus_texts = [item["q"] + " " + item["a"] for item in knowledge_base]
tfidf = TfidfVectorizer(stop_words="english")
tfidf.fit(corpus_texts)
idf_weights = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))


def sentence_vector(text):
    """Weighted average of word vectors -- rarer/more informative words count more."""
    words = re.findall(r"[a-z]+", text.lower())
    vecs, weights = [], []
    for w in words:
        if w in word_vectors:
            vecs.append(word_vectors[w])
            weights.append(idf_weights.get(w, 1.0))  # default weight if word not in our vocab
    if not vecs:
        return np.zeros(word_vectors.vector_size)
    vecs = np.array(vecs)
    weights = np.array(weights).reshape(-1, 1)
    return (vecs * weights).sum(axis=0) / weights.sum()


def cosine_sim(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


corpus_vectors = [sentence_vector(item["q"] + " " + item["a"]) for item in knowledge_base]


def retrieve(user_question, top_k=1):
    query_vec = sentence_vector(user_question)
    scores = [cosine_sim(query_vec, kb_vec) for kb_vec in corpus_vectors]
    best_idx = np.argsort(scores)[::-1][:top_k]
    return [(knowledge_base[i], scores[i]) for i in best_idx]


def generate(retrieved_item, score):
    """Placeholder for the LLM call -- see rag_demo.py notes."""
    if score < 0.6:
        return "I'm not confident I know this one -- try rephrasing, or contact student services directly."
    return f"[{retrieved_item['topic']}] {retrieved_item['a']}"


def chatbot(user_question):
    best_item, score = retrieve(user_question)[0]
    answer = generate(best_item, score)
    return answer, score


# ---- 3. Same test questions as the TF-IDF version, for direct comparison ----
demo_questions = [
    "how do i submit my application",
    "when do i need to pay fees",
    "is there a place to hang out on campus",
    "can I change my major",
    "forgot my email password, what do i do",
    "what's the weather like today",  # deliberately off-topic
]

if __name__ == "__main__":
    print("\n=== RAG Chatbot Demo v2 (embeddings) ===\n")
    for q in demo_questions:
        answer, score = chatbot(q)
        print(f"Q: {q}")
        print(f"A: {answer}  (confidence: {score:.2f})\n")
