# Customer Review Analysis App

This is a **Streamlit-based application** for analyzing customer reviews. It provides three main features:

1. **Sentiment Prediction**: Predict the sentiment of a user-input review using Logistic Regression, Naive Bayes, or a Neural Network model.  
2. **Negative Reviews Summarization**: Summarize negative reviews for each product from preloaded data.  
3. **Top Products Insights**: Show top products based on the number of positive or negative reviews and display a short summary for each product.

---

## Features

### 1️⃣ Sentiment Prediction
- Enter a review in the text area.
- Choose a model:
  - Logistic Regression
  - Naive Bayes
  - ANN (Neural Network)
- Get the predicted sentiment (Positive, Negative, or Neutral) instantly.
- Keep track of prediction history in the session.

### 2️⃣ Negative Reviews Summarization
- Preloaded dataset contains customer reviews for multiple products.
- Select a product to see the **main problems reported by customers**.
- Summarization is done using a **T5-small Transformer model** for efficiency on CPU.
- Supports hierarchical summarization for long texts.

### 3️⃣ Top Products Insights
- Select sentiment (Positive / Negative) and number of top products.
- Displays the top products based on review counts.
- Shows a **short summary** of reviews for each product using the Transformer summarizer.

---

## Dataset

- Place your CSV file with customer reviews in the `data/` folder and name it `dataForSammary.csv`.  
- The CSV should have at least the following columns:
  - `product_name`
  - `Final Review`
  - `Sentiment` (values: positive / negative / neutral)

---


## Installation

1. Clone the repository:

```bash
# Clone the repository
git clone https://github.com/AhmedByte/Customer-Review.git
cd Customer-Review
```

# Create and activate a Python environment (recommended)
```bash
conda create -n review_app python=3.10
conda activate review_app
```
# Install required packages
```bash
pip install -r requirements.txt
```
# Run the Streamlit app
```bash
streamlit run app.py
```



Notes

The app uses T5-small Transformer for summarization; it runs on CPU and may be slower for very large datasets.

Sentiment prediction models include Logistic Regression, Naive Bayes, and ANN (Neural Network).

Make sure to use Python 3.10 for compatibility with all libraries.