import streamlit as st
import joblib
import re
import pandas as pd
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import os

from review_summarizer import get_products, summarize_negative_reviews, summarize_product
from topn import get_top_products
from review_summarizer import summarize_product


# --- NLTK setup ---
nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

def download_nltk_resources():
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords', download_dir=nltk_data_dir)
    try:
        WordNetLemmatizer().lemmatize('test')
    except LookupError:
        nltk.download('wordnet', download_dir=nltk_data_dir)

download_nltk_resources()

# --- Load models & vectorizer ---
vectorizer = joblib.load("./models/sentiment_tfidf_vectorizer.pkl")
log_model = joblib.load("./models/sentiment_logisticRegression_model.pkl")
bayes_model = joblib.load("./models/sentiment_naiveBayes_model.pkl")
ann_model = load_model("./models/nn_sentiment_model.h5")

# --- Preprocessing ---
def process_review(text):
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned)

# --- Streamlit UI ---
st.title("Customer Review App")

tab1, tab2, tab3 = st.tabs([
    "Sentiment Prediction",
    "Negative Reviews Summarization",
    "Top Products Insights"
])

# --------------------- TAB 1: Sentiment Prediction ---------------------
with tab1:
    st.subheader("Enter a review to predict sentiment")
    user_input = st.text_area("Review Text")
    model_choice = st.selectbox("Select Model", ["Logistic Regression", "Naive Bayes", "ANN (Neural Network)"])

    if 'history' not in st.session_state:
        st.session_state.history = []

    def clear_history():
        st.session_state.history = []

    if st.button("Predict"):
        cleaned_text = process_review(user_input)
        vectorized_input = vectorizer.transform([cleaned_text])

        if model_choice == "Logistic Regression":
            prediction = log_model.predict(vectorized_input)[0]
        elif model_choice == "Naive Bayes":
            prediction = bayes_model.predict(vectorized_input)[0]
        elif model_choice == "ANN (Neural Network)":
            input_array = vectorized_input.toarray().astype('float32')
            y_pred_prob = ann_model.predict(input_array)
            prediction = y_pred_prob.argmax(axis=1)[0]

        label_map = {0: "Negative 😞", 1: "Positive 😊", 2: "Neutral 😐"}
        label = label_map[prediction]

        st.success(f"Prediction: {label}")
        st.session_state.history.append((model_choice, user_input, label))

    if st.session_state.history:
        st.subheader("Prediction History")
        history_df = pd.DataFrame(st.session_state.history, columns=["Model","Review", "Prediction"])
        history_df.index = history_df.index + 1
        st.write(history_df)

    st.button("Clear History", on_click=clear_history)

# --------------------- TAB 2: Negative Reviews Summarization ---------------------
with tab2:
    st.subheader("Negative Reviews Analysis (Preloaded Data)")

    products = get_products()
    selected_product = st.selectbox(
        "Select Product",
        products
    )

    if st.button("Summarize Negative Reviews"):
        with st.spinner("Analyzing customer complaints..."):
            summary = summarize_negative_reviews(selected_product)

        st.subheader("Main Problems Reported by Customers")
        st.write(summary)


# --------------------- TAB 3: Top Products Insights ---------------------
with tab3:
    st.subheader("Top Products Insights")

    sentiment = st.selectbox("Select Sentiment", ["positive", "negative"])
    top_n = st.slider("Number of Products", 1, 10, 5)

    if st.button("Show Top Products", key="show_top_products"):
        top_products = get_top_products(sentiment, top_n)

        if not top_products:
            st.warning("No products found for this sentiment.")
        else:
            for i, product in enumerate(top_products, start=1):
                st.markdown(f"### {i}. {product}")
                summary = summarize_product(product, sentiment)
                st.write(summary)
                st.divider()