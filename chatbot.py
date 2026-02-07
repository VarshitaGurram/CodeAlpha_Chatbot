import streamlit as st
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Chatbot", layout="wide")

# ---------- Fixed Theme Styling ----------
st.markdown("""
<style>

/* Force consistent look regardless of Streamlit theme */
html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #141E30, #243B55) !important;
    color: white !important;
}

/* Main Chat Card */
.main-container {
    max-width: 850px;
    margin: auto;
    margin-top: 40px;
    background: rgba(255,255,255,0.08);
    padding: 30px;
    border-radius: 20px;
    backdrop-filter: blur(15px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}

/* Chat Box */
.chat-box {
    max-height: 420px;
    overflow-y: auto;
    margin-bottom: 20px;
    padding-right: 5px;
}

/* User Bubble */
.user-msg {
    background: #4e73df;
    padding: 12px 18px;
    border-radius: 20px;
    margin: 10px 0;
    text-align: right;
    color: white;
    font-size: 15px;
}

/* Bot Bubble */
.bot-msg {
    background: rgba(255,255,255,0.15);
    padding: 12px 18px;
    border-radius: 20px;
    margin: 10px 0;
    text-align: left;
    color: white;
    font-size: 15px;
}

/* Confidence Text */
.confidence {
    font-size: 12px;
    opacity: 0.85;
    margin-top: 5px;
}

/* Input Styling */
input {
    background-color: rgba(255,255,255,0.15) !important;
    color: white !important;
}

button {
    background-color: #4e73df !important;
    color: white !important;
    border-radius: 10px !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-container'>", unsafe_allow_html=True)

st.markdown("## 🤖 AI FAQ Chatbot")
st.write("Ask anything about Artificial Intelligence")

# ---------- Knowledge Base ----------
faq_data = {
    "what is artificial intelligence":
        "Artificial Intelligence is the simulation of human intelligence by machines.",
    "what is machine learning":
        "Machine Learning enables systems to learn from data.",
    "what is deep learning":
        "Deep Learning uses neural networks to model complex patterns.",
    "what is nlp":
        "Natural Language Processing helps machines understand human language.",
    "what is python used for in ai":
        "Python is widely used in AI for machine learning and data analysis.",
    "what is computer vision":
        "Computer Vision allows machines to interpret visual information.",
    "what is data science":
        "Data Science combines programming and statistics to extract insights."
}

questions = list(faq_data.keys())
answers = list(faq_data.values())

# ---------- Preprocessing ----------
def preprocess(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    return text

processed_questions = [preprocess(q) for q in questions]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_questions)

# ---------- Session ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- Chat Display ----------
st.markdown("<div class='chat-box'>", unsafe_allow_html=True)

for msg in st.session_state.chat_history:
    if msg[0] == "user":
        st.markdown(f"<div class='user-msg'>{msg[1]}</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div class='bot-msg'>{msg[1]}"
            f"<div class='confidence'>🔎 Confidence: {msg[2]}%</div></div>",
            unsafe_allow_html=True
        )

st.markdown("</div>", unsafe_allow_html=True)

# ---------- Input ----------
col1, col2 = st.columns([4,1])

with col1:
    user_input = st.text_input("Type your question...", label_visibility="collapsed")

with col2:
    send = st.button("Send 🚀")

if send:
    if user_input.strip():
        processed_input = preprocess(user_input)
        user_vector = vectorizer.transform([processed_input])

        similarity = cosine_similarity(user_vector, tfidf_matrix)
        index = np.argmax(similarity)
        score = similarity[0][index]

        if score < 0.2:
            response = "I'm not sure about that. Please ask something related to AI topics."
        else:
            response = answers[index]

        confidence = round(score * 100, 2)

        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", response, confidence))

# ---------- Clear Button ----------
if st.button("🗑 Clear Chat"):
    st.session_state.chat_history = []

st.markdown("</div>", unsafe_allow_html=True)
