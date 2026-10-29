
from pathlib import Path

import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).with_name("emails.csv")


# ---------------- LOAD DATA & TRAIN MODEL ----------------

@st.cache_resource
def load_and_train():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "emails.csv not found. Keep it in the same folder as app.py."
        )

    data = pd.read_csv(DATA_PATH)

    if not {"label", "text"}.issubset(data.columns):
        raise ValueError("CSV must contain 'label' and 'text' columns.")

    data = data.dropna(subset=["label", "text"]).copy()
    data["label"] = data["label"].astype(str).str.lower().str.strip()
    data["text"] = data["text"].astype(str).str.strip()

    data = data[data["label"].isin(["ham", "spam"])]
    data = data[data["text"].str.len() > 0]
    data = data.drop_duplicates(subset=["text"])

    if data["label"].nunique() != 2:
        raise ValueError("Dataset must contain both ham and spam labels.")

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"],
        data["label"],
        test_size=0.25,
        random_state=42,
        stratify=data["label"],
    )

    def make_model():
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
            )),
            ("classifier", ComplementNB(alpha=0.5)),
        ])

    # Train on training data only for honest test-set evaluation.
    evaluation_model = make_model()
    evaluation_model.fit(X_train, y_train)
    test_predictions = evaluation_model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, test_predictions),
        "precision": precision_score(
            y_test, test_predictions, pos_label="spam", zero_division=0
        ),
        "recall": recall_score(
            y_test, test_predictions, pos_label="spam", zero_division=0
        ),
        "f1": f1_score(
            y_test, test_predictions, pos_label="spam", zero_division=0
        ),
        "matrix": confusion_matrix(
            y_test, test_predictions, labels=["ham", "spam"]
        ),
        "report": classification_report(
            y_test,
            test_predictions,
            labels=["ham", "spam"],
            output_dict=True,
            zero_division=0,
        ),
        "test_count": len(X_test),
        "train_count": len(X_train),
        "ham_count": int((data["label"] == "ham").sum()),
        "spam_count": int((data["label"] == "spam").sum()),
    }

    # Final prediction model trained on all available labelled data.
    final_model = make_model()
    final_model.fit(data["text"], data["label"])

    return final_model, metrics, len(data)


# ---------------- SESSION STATE ----------------

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# ---------------- SIDEBAR ----------------

st.sidebar.title("📧 EmailGuard")
st.sidebar.caption("Email Spam Classification System")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home & Predict",
        "📊 Model Performance",
        "🕘 Prediction History",
        "ℹ️ About Project",
    ],
)

st.sidebar.divider()
st.sidebar.markdown("### 🧠 Model Details")
st.sidebar.write("**Algorithm:** ComplementNB")
st.sidebar.write("**Features:** TF-IDF")
st.sidebar.write("**N-grams:** 1–2")
st.sidebar.write("**Dataset:** emails.csv")
st.sidebar.info(
    "Educational demo only. A HAM prediction does not guarantee "
    "that an email or its links are safe."
)


# ---------------- LOAD MODEL ----------------

try:
    model, metrics, total_emails = load_and_train()
except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
    st.error(str(error))
    st.stop()


# ---------------- HOME & PREDICT ----------------

if page == "🏠 Home & Predict":
    st.title("📧 Email Spam Classifier")
    st.write(
        "Check whether an email is classified as spam or normal email."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset Emails", f"{total_emails:,}")
    c2.metric("Training Emails", f"{metrics['train_count']:,}")
    c3.metric("Test Emails", f"{metrics['test_count']:,}")

    st.divider()

    email_text = st.text_area(
        "✉️ Paste Email Content",
        height=220,
        placeholder="Paste the email subject and body here...",
        key="email_input",
    )

    if st.button(
        "🔍 Check Spam",
        type="primary",
        use_container_width=True,
    ):
        if not email_text.strip():
            st.warning("Please enter some email text first.")
        else:
            prediction = model.predict([email_text])[0]
            probabilities = model.predict_proba([email_text])[0]
            classes = model.named_steps["classifier"].classes_
            scores = dict(zip(classes, probabilities))

            if prediction == "spam":
                st.error("🚨 SPAM — This email may be unwanted or suspicious.")
            else:
                st.success("✅ HAM — The model classifies this as normal email.")

            st.caption(
                "Model predictions are not guarantees. Check sender details "
                "and links independently."
            )

            left, right = st.columns(2)
            left.metric("Predicted Class", prediction.upper())
            right.metric(
                "Model Score for Prediction",
                f"{scores[prediction]:.2%}",
            )

            st.subheader("Class Scores")
            for label in ["ham", "spam"]:
                if label in scores:
                    st.write(f"**{label.upper()}** — {scores[label]:.2%}")
                    st.progress(float(scores[label]))

            # Keep session history, including prediction and scores.
            st.session_state.prediction_history.append({
                "Email Preview": " ".join(email_text.split())[:160],
                "Prediction": prediction.upper(),
                "HAM Score (%)": round(scores.get("ham", 0) * 100, 2),
                "SPAM Score (%)": round(scores.get("spam", 0) * 100, 2),
            })

    st.divider()
    st.subheader("🧪 Try Example Emails")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Normal email example**")
        st.code(
            "Hi Rahul, our meeting is scheduled for tomorrow at 10 AM. "
            "Please bring the project report."
        )

    with col2:
        st.markdown("**Suspicious email example**")
        st.code(
            "URGENT! You have won a lottery prize. Claim your FREE cash "
            "now by clicking this link immediately."
        )


# ---------------- MODEL PERFORMANCE ----------------

elif page == "📊 Model Performance":
    st.title("📊 Model Performance Dashboard")
    st.write(
        "These metrics are calculated on a held-out test set, "
        "not on the same data used to train the evaluation model."
    )

    a, b, c, d = st.columns(4)
    a.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    b.metric("Spam Precision", f"{metrics['precision']:.2%}")
    c.metric("Spam Recall", f"{metrics['recall']:.2%}")
    d.metric("Spam F1-score", f"{metrics['f1']:.2%}")

    st.caption(
        f"Train samples: {metrics['train_count']:,} | "
        f"Test samples: {metrics['test_count']:,} | "
        "Split: 75% / 25%, stratified, random_state=42"
    )

    st.divider()
    st.subheader("Confusion Matrix")
    st.caption("Rows = actual label; columns = predicted label.")

    cm = metrics["matrix"]
    cm_df = pd.DataFrame(
        cm,
        index=["Actual HAM", "Actual SPAM"],
        columns=["Predicted HAM", "Predicted SPAM"],
    )
    st.dataframe(cm_df, use_container_width=True)

    st.markdown(
        f"- Correct HAM predictions: **{cm[0, 0]}**\n"
        f"- HAM incorrectly flagged as SPAM: **{cm[0, 1]}**\n"
        f"- SPAM incorrectly classified as HAM: **{cm[1, 0]}**\n"
        f"- Correct SPAM predictions: **{cm[1, 1]}**"
    )

    st.divider()
    st.subheader("Classification Report")

    report_df = pd.DataFrame(metrics["report"]).T
    st.dataframe(
        report_df.round(3),
        use_container_width=True,
    )

    st.info(
        "Spam recall measures how many actual spam emails were detected. "
        "Spam precision measures how many emails flagged as spam really "
        "were spam in the labelled test set. Accuracy alone can be misleading."
    )


# ---------------- PREDICTION HISTORY ----------------

elif page == "🕘 Prediction History":
    st.title("🕘 Prediction History")
    st.write(
        "Predictions made during this running app session appear here."
    )

    history = st.session_state.prediction_history

    if not history:
        st.info(
            "No predictions recorded yet. Go to Home & Predict and test an email."
        )
    else:
        history_df = pd.DataFrame(history)
        st.metric("Predictions This Session", len(history_df))

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = history_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Download History as CSV",
            data=csv_data,
            file_name="prediction_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.prediction_history = []
            st.rerun()

    st.caption(
        "History is stored in the current Streamlit session. "
        "It is not permanent storage and may be lost when the session resets."
    )


# ---------------- ABOUT ----------------

elif page == "ℹ️ About Project":
    st.title("ℹ️ About This Project")

    st.markdown("""
    ### Objective
    Classify email text into two categories:

    - **HAM:** labelled as normal email in the dataset.
    - **SPAM:** labelled as spam in the dataset.

    ### Technologies
    - Python
    - Pandas
    - Scikit-learn
    - TF-IDF text features
    - Complement Naive Bayes
    - Streamlit

    ### Workflow
    1. Load and validate labelled email data.
    2. Split data into training and test sets.
    3. Evaluate the model on held-out test data.
    4. Train the final prediction model on all available labelled data.
    5. Predict new email text and show class scores.

    ### Limitations
    - Dataset quality and age affect performance.
    - New spam styles may not be recognized.
    - A prediction score is not a guarantee of safety.
    - This demo does not inspect URLs, attachments, or sender authenticity.
    """)

    st.info(
        "Do not use this educational model as the only protection "
        "for real email accounts."
    )

# ---------------- FOOTER ----------------

st.sidebar.caption("Email Spam Classification | Python + Streamlit")
