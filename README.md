# Legal Document Classification AI

This project is an NLP-based machine learning application that classifies legal documents into different categories such as Civil, Criminal, Tax, Service, and Constitutional.  
The system supports both text input and PDF upload and provides real-time predictions using a trained ML model deployed with Streamlit.

Live Website:  
https://legal-ai-classifier-7j9wxser45ft5nfsy5sjgc.streamlit.app/

---

## Features

- Multi-class legal document classification
- Supports text input and PDF upload
- TF-IDF based feature extraction
- LinearSVC machine learning model
- Streamlit web interface
- Real-time prediction
- Deployed on Streamlit Cloud

---

## Categories

The model classifies documents into:

- Civil
- Criminal
- Tax
- Service
- Constitutional

---

## Tech Stack

- Python
- Scikit-learn
- NLP (TF-IDF)
- Streamlit
- PyPDF2
- Joblib
- NumPy
- Pandas

---

## Project Workflow

1. Load dataset  
2. Text preprocessing  
3. TF-IDF vectorization  
4. Train LinearSVC model  
5. Save model using joblib  
6. Build Streamlit UI  
7. Add PDF text extraction  
8. Deploy on Streamlit Cloud  

---

## How to Run Locally

Clone repository


git clone https://github.com/harry20021025/legal-ai-classifier.git

cd legal-ai-classifier


Install dependencies


pip install -r requirements.txt


Run app


streamlit run app.py


---

## Project Files


app.py
legal_svm_model.pkl
legal_vectorizer.pkl
label_encoder.pkl
requirements.txt
README.md


---

## Live Demo

You can try the project here:

https://legal-ai-classifier-7j9wxser45ft5nfsy5sjgc.streamlit.app/

---

## Author

Hariom Dixit  
LinkedIn: https://www.linkedin.com/in/hariom-dixit-2b522a27a/  
GitHub: https://github.com/harry20021025
