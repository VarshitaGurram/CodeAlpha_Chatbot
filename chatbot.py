import streamlit as st
import json
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="CS AI Assistant", page_icon="🤖", layout="centered")

# ---------------- ADAPTIVE THEME CSS ----------------
st.markdown("""
<style>

/* Adaptive background */
.stApp {
    background: linear-gradient(
        135deg,
        var(--background-color),
        var(--secondary-background-color)
    );
    background-attachment: fixed;
}

/* Chat wrapper */
.chat-wrapper {
    max-width: 800px;
    margin: auto;
    padding: 20px;
}

/* User bubble (theme aware) */
.user-msg {
    background: var(--primary-color);
    color: white;
    padding: 12px 18px;
    border-radius: 20px;
    margin: 10px 0;
    width: fit-content;
    margin-left: auto;
}

/* Bot bubble */
.bot-msg {
    background: var(--secondary-background-color);
    color: var(--text-color);
    padding: 12px 18px;
    border-radius: 20px;
    margin: 10px 0;
    width: fit-content;
}

/* Buttons */
.stButton>button {
    border-radius: 12px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

st.title("🤖 Computer Science AI Assistant")
st.caption("Adaptive Theme + Semantic Matching")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------- LOAD FAQ ----------------
with open("faqs.json", "r") as f:
    data = json.load(f)

questions = [item["question"] for item in data]
answers = [item["answer"] for item in data]

question_embeddings = model.encode(questions)

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

if st.button("🧹 Clear Chat"):
    st.session_state.history = []

# ---------------- INPUT ----------------
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask your Computer Science question...")
    submit = st.form_submit_button("Send")

if submit and user_input:
    typing = st.empty()
    typing.markdown("🤖 Typing...")
    time.sleep(0.8)

    user_embedding = model.encode([user_input])
    similarities = cosine_similarity(user_embedding, question_embeddings)
    index = np.argmax(similarities)
    score = similarities[0][index]

    if score > 0.45:
        response = answers[index]
    else:
        response = "I couldn't find a strong match. Try rephrasing your question."

    typing.empty()

    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", response))

# ---------------- DISPLAY CHAT ----------------
st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)

for sender, msg in st.session_state.history:
    if sender == "You":
        st.markdown(f'<div class="user-msg">{msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">{msg}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
