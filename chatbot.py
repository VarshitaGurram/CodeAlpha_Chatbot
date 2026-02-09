import streamlit as st
import json
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CS Engineers AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

# ---------------- PREMIUM FIXED GRADIENT UI ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #141E30, #243B55);
    background-attachment: fixed;
}

/* Chat Wrapper */
.chat-wrapper {
    max-width: 800px;
    margin: auto;
    padding: 20px;
}

/* User Bubble */
.user-msg {
    background: linear-gradient(135deg, #00c6ff, #0072ff);
    padding: 12px 18px;
    border-radius: 20px;
    color: white;
    margin: 10px 0;
    width: fit-content;
    margin-left: auto;
}

/* Bot Bubble */
.bot-msg {
    background: rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
    padding: 12px 18px;
    border-radius: 20px;
    color: white;
    margin: 10px 0;
    width: fit-content;
}

/* Buttons */
.stButton>button {
    border-radius: 15px;
    background: linear-gradient(135deg, #00c6ff, #0072ff);
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Computer Science AI Assistant")
st.caption("Semantic FAQ Chatbot for Engineers")

# ---------------- LOAD MODEL (CACHED) ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------- SIDEBAR UPLOAD ----------------
st.sidebar.header("📂 Upload FAQ File (Optional)")
uploaded_file = st.sidebar.file_uploader("Upload JSON FAQ file", type=["json"])

if uploaded_file:
    data = json.load(uploaded_file)
else:
    with open("faqs.json", "r") as f:
        data = json.load(f)

questions = [item["question"] for item in data]
answers = [item["answer"] for item in data]

# Create embeddings
question_embeddings = model.encode(questions)

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- CLEAR BUTTON ----------------
if st.button("🧹 Clear Chat"):
    st.session_state.history = []

# ---------------- INPUT FORM ----------------
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask your Computer Science question...")
    submit = st.form_submit_button("Send")

if submit and user_input:

    # Typing animation
    typing_placeholder = st.empty()
    typing_placeholder.markdown("🤖 Typing...")
    time.sleep(1)

    # Semantic Matching
    user_embedding = model.encode([user_input])
    similarities = cosine_similarity(user_embedding, question_embeddings)
    index = np.argmax(similarities)
    score = similarities[0][index]

    if score > 0.45:
        response = answers[index]
    else:
        response = "I couldn't find a close FAQ match. Try rephrasing your question more clearly."

    typing_placeholder.empty()

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
