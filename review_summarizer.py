import pandas as pd
from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="./models/t5-small",
    tokenizer="models/t5-small",
    framework="pt"
)

DF = pd.read_csv("data/dataForSammary.csv")

DF['Sentiment'] = DF['Sentiment'].str.lower()
DF['Final Review'] = DF['Final Review'].astype(str)

def get_products():
    """ negative only products list """
    neg_df = DF[DF['Sentiment'] == "negative"]
    return sorted(neg_df['product_name'].unique())


def chunk_text(text, max_words=400):
    words = text.split()
    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
    ]

def summarize_negative_reviews(product_name):

    neg_reviews = DF[
        (DF['product_name'] == product_name) &
        (DF['Sentiment'] == "negative")
    ]['Final Review']

    if neg_reviews.empty:
        return "No negative reviews found for this product."

    combined_text = " ".join(neg_reviews.tolist())

    chunks = chunk_text(combined_text)

    summaries = []
    for chunk in chunks:
        out = summarizer(
            chunk,
            max_length=60,
            min_length=25,
            do_sample=False
        )
        summaries.append(out[0]['summary_text'])

    if len(summaries) > 1:
        final_text = " ".join(summaries)
        final_summary = summarizer(
            final_text,
            max_length=80,
            min_length=40,
            do_sample=False
        )[0]['summary_text']
        return final_summary

    return summaries[0]

def summarize_product(product_name, sentiment="negative"):
    """
    Generate a short summary for a product based on sentiment
    (positive / negative)
    """

    reviews = DF[
        (DF['product_name'] == product_name) &
        (DF['Sentiment'] == sentiment)
    ]['Final Review']

    if reviews.empty:
        return "No reviews available."

    combined_text = " ".join(reviews.tolist()[:10])

    output = summarizer(
        combined_text,
        max_length=50,
        min_length=20,
        do_sample=False
    )

    return output[0]['summary_text']
