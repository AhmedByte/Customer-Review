import pandas as pd
from review_summarizer import DF

def get_top_products(sentiment="negative", top_n=5):
    """
    ترجع قائمة بأعلى المنتجات حسب عدد الريفيوهات ل sentiment معين
    """
    df = DF[DF['Sentiment'] == sentiment]

    top_products = (
        df.groupby('product_name')
        .size()
        .sort_values(ascending=False)
        .head(top_n)
        .index
        .tolist()
    )

    return top_products
