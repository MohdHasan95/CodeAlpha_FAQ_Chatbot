import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from faq_data import FAQS

st.set_page_config(page_title="CodeAlpha FAQ Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 CodeAlpha Internship FAQ Chatbot")
st.caption("Ask questions about the CodeAlpha AI internship, tasks, and submission process.")

# Explicit task-number handling prevents Task 1 and Task 4 questions from being confused.
TASK_ANSWERS = {
    1: "Task 1 is the Language Translation Tool. It involves building a user interface, selecting source and target languages, using a translation API, and displaying translated text. Copy and text-to-speech features are optional.",
    2: "Task 2 is the Chatbot for FAQs. It involves collecting FAQs and answers, preprocessing text, matching a user's question using cosine similarity or intent matching, and displaying the best matching answer. A simple chat interface is optional.",
    3: "Task 3 is Music Generation with AI. It involves working with MIDI data, preprocessing note sequences, training a model such as an RNN/LSTM or GAN, and converting generated sequences to MIDI or audio.",
    4: "Task 4 is Object Detection and Tracking. It involves using video input, applying a pre-trained model such as YOLO or Faster R-CNN, drawing bounding boxes around detected objects, and tracking objects with IDs."
}

@st.cache_resource
def build_search_index():
    questions = [item["question"] for item in FAQS]
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))
    question_vectors = vectorizer.fit_transform(questions)
    return vectorizer, question_vectors

vectorizer, question_vectors = build_search_index()

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! I can answer FAQs about the CodeAlpha AI internship. What would you like to know?"
    }]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input("Type your question here...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    task_match = re.search(r"\btask\s*([1-4])\b", user_question, flags=re.IGNORECASE)
    if task_match:
        task_number = int(task_match.group(1))
        answer = TASK_ANSWERS[task_number]
    else:
        query_vector = vectorizer.transform([user_question])
        scores = cosine_similarity(query_vector, question_vectors)[0]
        best_index = scores.argmax()
        best_score = float(scores[best_index])

        if best_score >= 0.18:
            answer = FAQS[best_index]["answer"]
        else:
            answer = (
                "Sorry, I couldn't find a reliable answer in my current FAQ collection. "
                "Please try asking about internship tasks, GitHub upload, LinkedIn posts, "
                "certificates, or submission."
            )

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

with st.sidebar:
    st.subheader("About this chatbot")
    st.write(
        "This chatbot uses TF-IDF text vectorization and cosine similarity "
        "to find the FAQ most similar to the user's query. Explicit task-number "
        "questions are handled directly to avoid confusing different tasks."
    )
    st.caption("The FAQ content is based on the CodeAlpha internship instruction PDF.")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
