import joblib
import pandas as pd
from textblob import TextBlob

# show all columns
pd.set_option('display.max_columns', None)


def predict_sentiment(text):
    # load the best model
    best_model = joblib.load("data/best_model.pkl")

    # load tfidf
    tfidf = joblib.load("data/tfidf.pkl")

    # prediction confidence
    pred_proba = best_model.predict_proba(tfidf.transform([text]))
    # print(pred_proba[0])
    # get max confidence
    max_proba = max(pred_proba[0])
    # print(max_proba)

    # 0-1 value to -1 to 1
    b = (max_proba - 0.5) * 2

    # find polarity
    polarity = TextBlob(text).sentiment.polarity

    # combine polarity and pred with weight
    weight = 0.5
    final_pred = (polarity * weight) + (b * (1 - weight))

    if final_pred < 0:
        result = 'Negative'
    else:
        result = 'Positive'

    return result
