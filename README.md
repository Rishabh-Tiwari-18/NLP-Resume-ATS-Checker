# NLP Resume ATS Checker

An NLP and Machine Learning based Resume ATS Checker that classifies resumes into different job categories using Natural Language Processing, TF-IDF Vectorization, and multiple Machine Learning algorithms.

---

# Project Overview

This project simulates an Applicant Tracking System (ATS) used by recruiters to automatically screen and classify resumes.

The system preprocesses resume text using NLP techniques, converts text into numerical vectors using TF-IDF, and predicts the most suitable job category using trained Machine Learning models.

The project also includes a Streamlit-based web application for interactive resume analysis and prediction.

---

# Features

- Resume category prediction
- NLP-based text preprocessing
- TF-IDF feature extraction
- Multi-model comparison
- Streamlit web interface
- Real-time prediction
- Saved ML models using Joblib

---

# Machine Learning Models Used

| Model | Accuracy |
|---|---|
| Logistic Regression | 84% |
| Random Forest | 77% |
| XGBoost | 90% |

---

# NLP Techniques Used

- Lowercasing
- Regex cleaning
- Stopword removal
- Lemmatization
- Tokenization
- TF-IDF Vectorization
- Unigram and Bigram feature extraction

---

# Tech Stack

## Programming Language
- Python

## Libraries & Frameworks
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- NLTK
- Streamlit
- Joblib

---

# Project Structure

```bash
NLP-Resume-ATS-Checker/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── dataset/
│   └── UpdatedResumeDataSet.csv
│
├── models/
│   ├── resume_classifier.pkl
│   ├── tfidf.pkl
│   └── label_encoder.pkl
│
├── utils/
│   └── preprocess.py
│
└── assets/
    └── banner.png
