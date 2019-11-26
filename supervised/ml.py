import glob
import os
import pickle
import re
from pathlib import Path

import pandas as pd
import unidecode
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split


def load_train_data():
    dataset = pd.read_csv(
        '..\\csv\\training.1600000.processed.noemoticon.csv',
        encoding='ISO-8859-1', header=None)
    X = dataset.iloc[:, 5].values
    X = pd.Series(X)
    y = dataset.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)
    return X_train, y_train, X_test, y_test


def load_train_data_2():
    dataset = pd.read_excel('../newTraining.xlsx')
    dataset = dataset[['sentiment', 'text']].dropna()
    X = dataset.iloc[:, 1].values
    X = pd.Series(X)
    y = dataset.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)
    return X_train, y_train, X_test, y_test

def preprocess(tweet):
    # Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)

    # Convert @username to __USERHANDLE
    tweet = re.sub('@[^\s]+', '__USERHANDLE', tweet)

    # Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

    # trim
    tweet = tweet.strip('\'"')

    # Repeating words like hellloooo
    repeat_char = re.compile(r"(.)\1+", re.IGNORECASE)
    tweet = repeat_char.sub(r"\1\1", tweet)

    # remove non ascii
    tweet = unidecode.unidecode(tweet)

    emoticons = [':-)', ':)', '(:', '(-:',
                 ':-D', ':D', 'X-D', 'XD', 'xD',
                 '<3', ':\*', ';-)', ';)', ';-D', ';D', '(;', '(-;',
                 ':-(', ':(', '(:', '(-:', ':,(',
                 ':\'(', ':"(', ':((', 'D:']
    for emoticon in emoticons:
        tweet = tweet.replace(emoticon, ' ')

    # Convert to lower case
    tweet = tweet.lower()

    return tweet


def preprocess_data(data):
    X_train, y_train, X_test, y_test = data
    X_train = [preprocess(tweet) for tweet in X_train]
    X_test = [preprocess(tweet) for tweet in X_test]
    return X_train, y_train, X_test, y_test


def fit_tfidf(X):
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.95, sublinear_tf=True, use_idf=True, ngram_range=(1, 2))
    vectorizer.fit(X)
    return vectorizer


def evaluate(y_pred, y_true):
    return metrics.accuracy_score(y_true, y_pred)


def confusion_matrix(y_pred, y_test):
    return metrics.confusion_matrix(y_test, y_pred)


def pickle_model(model, fname):
    pickle.dump(model, open(fname, 'wb'))


def unpickle_model(fname):
    return pickle.load(open(fname, 'rb'))


def predict(model, X):
    return model.predict(X)


def roc(y_test, y_score):
    return roc_curve(y_test, y_score, pos_label=0)

# Logistic Regression

def train_lr(X_train, y_train, **kwargs):
    model = LogisticRegression(**kwargs)
    model.fit(X_train, y_train)
    return model


# GBDT

def train_gbdt(X_train, y_train, n_est=100, lr=0.1, feats=50, **kwargs):
    gbdt = GradientBoostingClassifier(
        n_estimators=n_est, learning_rate=lr,
        max_features=feats, random_state=0,
        **kwargs
    )
    gbdt.fit(X_train, y_train)
    return gbdt


def predict_test(
    vectorizer, model, out_path: Path,
    data_path=Path('C:\\Users\\prach\\Desktop\\Political-Opinion-Mining\\all\\')
):
    out_path.mkdir(exist_ok=True)
    for fl in glob.glob(str(data_path / '*.csv')):
        data = pd.read_csv(fl, encoding='utf-8-sig', header=None)
        tweets = data.iloc[:, 6]
        tweets = [preprocess(tweet) for tweet in tweets]
        predictions = predict(model, vectorizer.transform(tweets))
        data.iloc[:, 6] = tweets
        data.insert(0, 'sentiment', predictions)
        data.to_csv(out_path / Path(fl).name, header=None)