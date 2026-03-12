import streamlit as st
import joblib
import re
import string
import PyPDF2


# ======================
# Load model
# ======================

model = joblib.load("legal_svm_model.pkl")
vectorizer = joblib.load("legal_vectorizer.pkl")
le = joblib.load("label_encoder.pkl")


# ======================
# Clean text
# ======================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"\d+", " ", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    return text


# ======================
# Read PDF
# ======================

def read_pdf(file):

    pdf = PyPDF2.PdfReader(file)

    text = ""

    for page in pdf.pages:
        text += page.extract_text()

    return text


# ======================
# UI
# ======================

st.title("Legal Document Classification AI")

st.write("Upload PDF OR paste text")


# -------- TEXT INPUT --------

text_input = st.text_area(
    "Enter text here",
    height=200
)


# -------- PDF INPUT --------

pdf_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)


# ======================
# Predict
# ======================

if st.button("Predict"):

    text = ""

    # If PDF uploaded
    if pdf_file is not None:

        text = read_pdf(pdf_file)

    # If text entered
    elif text_input != "":

        text = text_input

    else:

        st.warning("Enter text or upload PDF")

    if text != "":

        clean = clean_text(text)

        vec = vectorizer.transform([clean])

        pred = model.predict(vec)[0]

        label = le.inverse_transform([pred])[0]

        st.success(f"Prediction: {label}")
