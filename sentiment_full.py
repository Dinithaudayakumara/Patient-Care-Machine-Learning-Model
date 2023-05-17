import gensim
import joblib
import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import re
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from textblob import TextBlob

# POS tagger dictionary
pos_dict = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}

# show all columns
pd.set_option('display.max_columns', None)

from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = re.sub('[^A-Za-z]+', ' ', text)
    text = text.strip()
    text = text.split()
    text = [word for word in text if word not in stopwords.words('english')]
    text = ' '.join(text)

    return text


def token_stop_pos(text):
    tags = pos_tag(word_tokenize(text))
    newlist = []
    for word, tag in tags:
        if word.lower() not in set(stopwords.words('english')):
            newlist.append(tuple([word, pos_dict.get(tag[0])]))
    return newlist


def lemmatize(pos_data):
    lemma_rew = " "
    for word, pos in pos_data:
        if not pos:
            lemma = word
            lemma_rew = lemma_rew + " " + lemma
        else:
            lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
            lemma_rew = lemma_rew + " " + lemma
    return lemma_rew


def pre_process(df):
    # drop null values
    df = df.dropna()

    # drop duplicates
    df = df.drop_duplicates()

    # keep only column_name, value
    df = df[['column_name', 'value']]

    # rename to selected_text	sentiment
    df = df.rename(columns={'column_name': 'sentiment', 'value': 'selected_text'})

    print(df.tail())

    df["selected_text"].isnull().sum()
    df["selected_text"].fillna("No content", inplace=True)

    return df


def depure_data(data):
    # Removing URLs with a regular expression
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    data = url_pattern.sub(r'', data)

    # Remove Emails
    data = re.sub('\S*@\S*\s?', '', data)

    # Remove new line characters
    data = re.sub('\s+', ' ', data)

    # Remove distracting single quotes
    data = re.sub("\'", "", data)

    return data


def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


def detokenize(text):
    return TreebankWordDetokenizer().detokenize(text)


def process_data():
    # load csv
    df = pd.read_csv("data/data.csv")

    print(df.tail())

    df = df.dropna()
    df = df.drop_duplicates()

    print("Cleaning text...")
    df['cleaned_text'] = df['text'].apply(lambda x: clean_text(x.lower()))
    print("Tokenizing text...")
    df['tokenized_text'] = df['cleaned_text'].apply(lambda x: word_tokenize(x))
    print("POS tagging text...")
    df['POS tagged'] = df['cleaned_text'].apply(token_stop_pos)
    print("Lemmatizing text...")
    df['Lemma'] = df['POS tagged'].apply(lemmatize)

    print("Done!")

    # to csv
    df.to_csv("data/train.csv", index=False)


def train_sentiment():
    # load csv
    df = pd.read_csv("data/train.csv")

    # drop null values
    df = df.dropna().drop_duplicates()

    # keep only column_name, value
    df = df[['Lemma', 'sentiment']]

    print(df.tail())

    df["Lemma"].isnull().sum()
    df["Lemma"].fillna("No content", inplace=True)

    temp = []
    # Splitting pd.Series to list
    data_to_list = df['Lemma'].values.tolist()
    for i in range(len(data_to_list)):
        temp.append(depure_data(data_to_list[i]))
    list(temp[:5])

    def sent_to_words(sentences):
        for sentence in sentences:
            yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    data_words = list(sent_to_words(temp))

    print(data_words[:10])

    data = []
    for i in range(len(data_words)):
        data.append(detokenize(data_words[i]))
    print(data[:5])

    tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)

    X = tfidf.fit_transform(df['Lemma'])
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    lr = LogisticRegression(solver='liblinear')
    mnb = MultinomialNB()
    sgd = SGDClassifier()
    svc = SVC(probability=True)
    rf = RandomForestClassifier()

    lr.fit(X_train, y_train)
    mnb.fit(X_train, y_train)
    sgd.fit(X_train, y_train)
    svc.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    preds_lr = lr.predict(X_test)
    preds_mnb = mnb.predict(X_test)
    preds_sgd = sgd.predict(X_test)
    preds_svc = svc.predict(X_test)
    preds_rf = rf.predict(X_test)

    # find the best model
    ac_lr = accuracy_score(preds_lr, y_test)
    ac_mnb = accuracy_score(preds_mnb, y_test)
    ac_sgd = accuracy_score(preds_sgd, y_test)
    ac_svc = accuracy_score(preds_svc, y_test)
    ac_rf = accuracy_score(preds_rf, y_test)

    print("Logistic Regression: ", ac_lr)
    print("Multinomial Naive Bayes: ", ac_mnb)
    print("SGD Classifier: ", ac_sgd)
    print("Support Vector Classifier: ", ac_svc)
    print("Random Forest Classifier: ", ac_rf)

    # find the best model
    models = [lr, mnb, sgd, svc, rf]
    accuracies = [ac_lr, ac_mnb, ac_sgd, ac_svc, ac_rf]
    best_model = models[accuracies.index(max(accuracies))]
    print("Best model: ", best_model)

    # save the best model
    joblib.dump(best_model, "data/best_model.pkl")

    # save tfidf
    joblib.dump(tfidf, "data/tfidf.pkl")

    print("-" * 50)

    print("Done Training")


def predict_sentiment(text):
    # load the best model
    best_model = joblib.load("data/best_model.pkl")

    # load tfidf
    tfidf = joblib.load("data/tfidf.pkl")

    pred = best_model.predict(tfidf.transform([text]))
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
    # print("polarity: ", polarity)

    # combine polarity and keras_pred with weight
    weight = 0.5
    final_pred = (polarity * weight) + (b * (1 - weight))
    # print("final_pred: ", final_pred)

    if final_pred < 0:
        result = 'Negative'
    else:
        result = 'Positive'

    return result
