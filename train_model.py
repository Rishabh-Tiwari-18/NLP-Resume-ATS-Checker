import pandas as pd
import numpy as np
import re
import nltk
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# NLTK setup

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


df = pd.read_csv("dataset/UpdatedResumeDataSet.csv")

print(df.head())
print("\nOriginal Dataset Shape:", df.shape)


df = df.dropna(subset=["Category", "Resume"])
df = df.drop_duplicates()
df["Resume"] = df["Resume"].astype(str)

print("\nAfter basic cleaning:", df.shape)


# Remove rare classes

min_samples = 5

df = df.groupby("Category").filter(lambda x: len(x) >= min_samples)

print("\nCategory distribution after filtering:\n")
print(df["Category"].value_counts())

print("\nFinal Dataset Shape:", df.shape)


# Safety check

if df.shape[0] == 0:
    print("Dataset empty after filtering. Reduce min_samples.")
    exit()


# Text cleaning

def clean_resume(text):
    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)

    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]

    return " ".join(words)


# Apply cleaning

df["cleaned_resume"] = df["Resume"].apply(clean_resume)

print("\nSample cleaned resume:\n")
print(df["cleaned_resume"].iloc[0][:500])


# Encode labels

label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["Category"])


# Features

X = df["cleaned_resume"]
y = df["label"]


# TF-IDF 

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_vec = tfidf.fit_transform(X)


# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X_vec,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# LOGISTIC REGRESSION

LogReg = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    solver="lbfgs"
)

LogReg.fit(X_train, y_train)


# Evaluation

y_pred = LogReg.predict(X_test)

print("\nAccuracy Logistic Regression:", accuracy_score(y_test, y_pred))

print("\nClassification Report Logistic Regression:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

RF.fit(X_train, y_train)

# Evaluation

y_pred_rf = RF.predict(X_test)

print("\nAccuracy Random Forest:", accuracy_score(y_test, y_pred_rf))

print("\nClassification Report Random Forest:\n")
print(classification_report(y_test, y_pred_rf, zero_division=0))


# XGBOOST

from xgboost import XGBClassifier

XGB = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    objective="multi:softmax",
    num_class=len(label_encoder.classes_),
    eval_metric="mlogloss",
    random_state=42
)

XGB.fit(X_train, y_train)

# Evaluation

y_pred_xgb = XGB.predict(X_test)

print("\nAccuracy XGBoost:", accuracy_score(y_test, y_pred_xgb))

print("\nClassification Report XGBoost:\n")
print(classification_report(y_test, y_pred_xgb, zero_division=0))
