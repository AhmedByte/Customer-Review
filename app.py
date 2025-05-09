import streamlit as st
import joblib
import re
import pandas as pd
import torch
import torch.nn as nn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load models
vectorizer = joblib.load(r"C:\Users\Ahmed\Desktop\NLP project\vectorizer.pkl")
log_model = joblib.load(r"C:\Users\Ahmed\Desktop\NLP project\logistic_regression_model.pkl")
bayes_model = joblib.load(r"C:\Users\Ahmed\Desktop\NLP project\naive_bayes_model.pkl")

# Define ANN Model
class ANNModel(nn.Module):
    def __init__(self, input_dim):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Initialize and load ANN model
input_dim = vectorizer.transform(["sample"]).shape[1]
ann_model = ANNModel(input_dim)
ann_model.load_state_dict(torch.load(r"C:\Users\Ahmed\Desktop\NLP project\ann_model.pth"))
ann_model.eval()

# Text preprocessing function
def process_review(text):
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned)

# Streamlit UI
st.title("Sentiment Prediction App")
st.write("Enter a review and choose a model for prediction.")

user_input = st.text_area("Review Text")

model_choice = st.selectbox("Select Model", ["Logistic Regression", "Naive Bayes", "ANN (Neural Network)"])

# Initialize session state for storing history
if 'history' not in st.session_state:
    st.session_state.history = []

# Function to clear history
def clear_history():
    st.session_state.history = []

# Predict sentiment
if st.button("Predict"):
    cleaned_text = process_review(user_input)
    vectorized_input = vectorizer.transform([cleaned_text])

    if model_choice == "Logistic Regression":
        prediction = log_model.predict(vectorized_input)[0]
    elif model_choice == "Naive Bayes":
        prediction = bayes_model.predict(vectorized_input)[0]
    elif model_choice == "ANN (Neural Network)":
        input_tensor = torch.tensor(vectorized_input.toarray(), dtype=torch.float32)
        with torch.no_grad():
            output = ann_model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()

    label = "Positive 😊" if prediction == 1 else "Negative 😞"
    st.success(f"Prediction: {label}")

    # Save the input and prediction to history
    st.session_state.history.append((user_input, label))

# Display history of predictions
if st.session_state.history:
    st.subheader("Review Prediction History")
    history_df = pd.DataFrame(st.session_state.history, columns=["Review", "Prediction"])
    st.write(history_df)

# Button to clear history
st.button("Clear History", on_click=clear_history)
