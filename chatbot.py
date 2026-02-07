import streamlit as st
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Premium AI Chatbot", layout="centered")

# ---------- Premium UI ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}
.chat-container {
    max-width: 800px;
    margin: auto;
}
.user-message {
    background: #4e73df;
    padding: 12px 18px;
    border-radius: 20px;
    margin: 8px 0;
    text-align: right;
    color: white;
}
.bot-message {
    background: rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
    padding: 12px 18px;
    border-radius: 20px;
    margin: 8px 0;
    text-align: left;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI FAQ Chatbot")
st.write("Ask anything related to Artificial Intelligence.")

# ---------- Knowledge Base ----------
faq_data = {
    "what is artificial intelligence":
        "Artificial Intelligence is the simulation of human intelligence by machines.",
    "what is machine learning":
        "Machine Learning is a subset of AI that enables systems to learn from data.",
    "what is deep learning":
        "Deep Learning is a type of machine learning using neural networks.",
    "what is nlp":
        "Natural Language Processing helps machines understand human language.",
    "what is python used for in ai":
        "Python is widely used in AI for data analysis, machine learning, and automation.",
    "what is computer vision":
        "Computer Vision allows machines to interpret and understand visual data.",
    "what is data science":
        "Data Science combines statistics and programming to extract insights from data."
}

questions = list(faq_data.keys())
answers = list(faq_data.values())

# ---------- Preprocessing ----------
def preprocess(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    return text

processed_questions = [preprocess(q) for q in questions]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_questions)

# ---------- Session State ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- Clear Chat ----------
if st.button("🗑 Clear Chat"):
    st.session_state.chat_history = []

# ---------- User Input ----------
user_input = st.text_input("Type your question")

if st.button("Send"):
    if user_input.strip():
        processed_input = preprocess(user_input)
        user_vector = vectorizer.transform([processed_input])

        similarity = cosine_similarity(user_vector, tfidf_matrix)
        index = np.argmax(similarity)
        score = similarity[0][index]

        if score < 0.2:
            response = "I’m not sure about that. Please ask something related to AI topics."
        else:
            response = answers[index]

        confidence = round(score * 100, 2)

        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", response, confidence))

# ---------- Display Chat ----------
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

for message in st.session_state.chat_history:
    if message[0] == "user":
        st.markdown(
            f"<div class='user-message'>{message[1]}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='bot-message'>{message[1]}<br><br>🔎 Confidence: {message[2]}%</div>",
            unsafe_allow_html=True
        )

st.markdown("</div>", unsafe_allow_html=True)
