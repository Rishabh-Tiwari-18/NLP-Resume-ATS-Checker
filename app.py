import streamlit as st
import joblib
import PyPDF2
import re


# Load Models


model = joblib.load("models/resume_classifier.pkl")

tfidf = joblib.load("models/tfidf.pkl")

label_encoder = joblib.load("models/label_encoder.pkl")


# Streamlit Config


st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)


# Custom CSS


st.markdown(
    '''
    <style>

    .main {
        background-color: #0E1117;
        color: white;
    }

    .stButton>button {
        background-color: #00ADB5;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }

    .stFileUploader {
        border: 2px dashed #00ADB5;
        padding: 20px;
        border-radius: 10px;
    }

    </style>
    ''',
    unsafe_allow_html=True
)


# Skills Database


skills_db = {

    "Data Science": [
        "python",
        "machine learning",
        "deep learning",
        "tensorflow",
        "sql",
        "pandas",
        "numpy",
        "power bi"
    ],

    "Web Development": [
        "html",
        "css",
        "javascript",
        "react",
        "node",
        "mongodb"
    ],

    "Android Development": [
        "java",
        "kotlin",
        "firebase",
        "android"
    ],

    "Cyber Security": [
        "kali linux",
        "wireshark",
        "penetration testing",
        "network security"
    ]
}


# Clean Resume


def clean_resume(text):

    text = text.lower()

    text = re.sub(r'http\\S+', ' ', text)

    text = re.sub(r'[^a-zA-Z ]', ' ', text)

    text = re.sub(r'\\s+', ' ', text)

    return text


# Extract PDF Text


def extract_text_from_pdf(pdf_file):

    text = ""

    pdf_reader = PyPDF2.PdfReader(pdf_file)

    for page in pdf_reader.pages:

        extracted = page.extract_text()

        if extracted:
            text += extracted

    return text


# ATS Score


def calculate_ats_score(text):

    keywords = [
        "python",
        "sql",
        "machine learning",
        "deep learning",
        "tensorflow",
        "communication",
        "leadership",
        "teamwork",
        "project",
        "internship"
    ]

    score = 0

    found_keywords = []

    for word in keywords:

        if word.lower() in text.lower():

            score += 10

            found_keywords.append(word)

    score = min(score, 100)

    return score, found_keywords


# Header


st.title("📄 AI Resume Screening System")

st.write(
    "Upload your resume and get AI-powered ATS analysis, "
    "skill extraction, and category prediction."
)


# File Upload


uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)


# Process Resume


if uploaded_file is not None:

    with st.spinner("Analyzing Resume..."):

        # Extract text
        resume_text = extract_text_from_pdf(uploaded_file)

        # Clean text
        cleaned_resume = clean_resume(resume_text)

        # Transform
        resume_vector = tfidf.transform([cleaned_resume])

        # Prediction
        prediction = model.predict(resume_vector)

        category = label_encoder.inverse_transform(prediction)[0]

        # ATS Score
        ats_score, found_keywords = calculate_ats_score(resume_text)

        # Extract Skills
        extracted_skills = []

        for skill in skills_db.get(category, []):

            if skill.lower() in resume_text.lower():

                extracted_skills.append(skill)

        # Missing Skills
        missing_skills = list(
            set(skills_db.get(category, [])) -
            set(extracted_skills)
        )

    
    # Results Layout
    

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🎯 Predicted Category")

        st.success(category)

        st.subheader("📊 ATS Score")

        st.progress(ats_score / 100)

        st.write(f"{ats_score}%")

    with col2:

        st.subheader("🛠 Detected Skills")

        st.write(extracted_skills)

        st.subheader("❌ Missing Skills")

        st.write(missing_skills)

    
    # Keywords
    

    st.subheader("🔑 Keywords Found")

    st.write(found_keywords)

    
    # Recommendations
    

    st.subheader("💡 Recommendations")

    if ats_score < 50:

        st.error(
            "Your resume needs improvement. Add more projects, "
            "technical skills, and industry keywords."
        )

    elif ats_score < 80:

        st.warning(
            "Good resume but can be improved with more relevant "
            "skills and keywords."
        )

    else:

        st.success(
            "Excellent ATS-friendly resume."
        )

    
    # Resume Text
    

    with st.expander("📄 Extracted Resume Text"):

        st.write(resume_text)