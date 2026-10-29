from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Email Spam Detector", page_icon="📧")
st.title("📧 Email Spam Detector")
st.write("Paste an email sessage to classify it as **Spam** or **Ham**.")

@st.cache_resource
def train_model():
    data = pd.read_csv(Path(__file__).with_name("emails.csv")).dropna(subset=["label", "text"])
    data["label"] = data["label"].str.lower().str.strip()
    data = data[data["label"].isin(["spam", "ham"])]
    model = Pipeline([("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
                      ("classifier", MultinomialNB())])
    model.fit(data["text"], data["label"])
    return model, len(data)

model, count = train_model()
st.caption(f"Demo model trained on {count} sample messages.")
message = st.text_area("Email message", height=160, placeholder="Congratulations! You have won a prize...")
if st.button("Check Email", type="primary"):
    if not message.strip():
        st.warning("Please enter an email message first.")
    elif model.predict([message])[0] == "spam":
        st.error("Prediction: SPAM — सावधगिरी बाळगा.")
    else:
        st.success("Prediction: HAM — सामान्य ई-मेल असण्याची शक्यता.")
st.info("Educational demo only. Predictions may be wrong; do not use for real security decisions.")
