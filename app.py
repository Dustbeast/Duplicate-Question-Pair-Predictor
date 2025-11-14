import streamlit as st
import pickle
import numpy as np
import helper

# -------------------------------
# Streamlit Page Configuration (MUST BE FIRST)
# -------------------------------
st.set_page_config(
    page_title="Duplicate Question Pair Detector",
    page_icon="🔍",
    layout="centered"
)

# -------------------------------
# Load Model & Vectorizer
# -------------------------------
@st.cache_resource
def load_model():
    model = pickle.load(open("xgb.pkl", "rb"))
    return model

model = load_model()

# -------------------------------
# Custom CSS
# -------------------------------
page_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.title {
    font-size: 42px;
    font-weight: 700;
    color: #2E86C1;
    text-align: center;
}
.subtitle {
    font-size: 18px;
    text-align: center;
    color: #555;
}
.result-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #F2F3F4;
    text-align: center;
    margin-top: 10px;
}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# -------------------------------
# UI Headings
# -------------------------------
st.markdown('<div class="title">Duplicate Question Pair Detection 🔍</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter two questions and the model will determine if they are duplicates.</div>', unsafe_allow_html=True)
st.write("")

# User Inputs
q1 = st.text_area("✏️ Enter Question 1")
q2 = st.text_area("✏️ Enter Question 2")

def predict_duplicate(question1, question2):
    query = helper.query_point_creator(question1, question2)
    prediction = model.predict(query)[0]
    proba = model.predict_proba(query)[0] if hasattr(model, "predict_proba") else None
    return prediction, proba

if st.button("🔎 Predict"):
    if q1.strip() == "" or q2.strip() == "":
        st.warning("Please enter both questions before predicting.")
    else:
        pred, proba = predict_duplicate(q1, q2)

        if pred == 1:
            st.success("✔ The questions are **Duplicate**.")
        else:
            st.error("✖ The questions are **Not Duplicate**.")

        if proba is not None:
            st.write("---")
            st.subheader("📊 Prediction Confidence:")
            st.write(f"Duplicate: **{proba[1]*100:.2f}%**")
            st.write(f"Not Duplicate: **{proba[0]*100:.2f}%**")

st.write("---")
st.caption("Built with ❤️ using Streamlit & Machine Learning")
