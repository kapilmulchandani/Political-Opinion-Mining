

import nltk
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pylab as pl
from nltk.stem import WordNetLemmatizer



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
    repeat_char = re.compile(r"(.)\1{1,}", re.IGNORECASE)
    tweet = repeat_char.sub(r"\1\1", tweet)

    # Emoticons
    emoticons = [
        ('__positive__', [':-)', ':)', '(:', '(-:', \
                          ':-D', ':D', 'X-D', 'XD', 'xD', \
                          '<3', ':\*', ';-)', ';)', ';-D', ';D', '(;', '(-;', ]), \
        ('__negative__', [':-(', ':(', '(:', '(-:', ':,(', \
                          ':\'(', ':"(', ':((', 'D:']), \
        ]

    def replace_parenthesis(arr):
        return [text.replace(')', '[)}\]]').replace('(', '[({\[]') for text in arr]

    def join_parenthesis(arr):
        return '(' + '|'.join(arr) + ')'

    emoticons_regex = [(repl, re.compile(join_parenthesis(replace_parenthesis(regx)))) for (repl, regx) in emoticons]

    for (repl, regx) in emoticons_regex:
        tweet = re.sub(regx, ' ' + repl + ' ', tweet)

    # Convert to lower case
    tweet = tweet.lower()

    return tweet

# Lemmatization of Tweets
def lemma(tweet):
    wordnet_lemmatizer = WordNetLemmatizer()
    punctuations="?:!.,;"
    sentence_words = nltk.word_tokenize(tweet)
    sentence_words_lemma = []
    for word in sentence_words:
        if word in punctuations:
            sentence_words.remove(word)
        else:
            sentence_words_lemma.append(wordnet_lemmatizer.lemmatize(word,pos="v"))
    return ' '.join(sentence_words_lemma)
# Stemming of Tweets

def stem(tweet):
    stemmer = nltk.stem.PorterStemmer()
    tweet_stem = ''
    words = [word if (word[0:2] == '__') else word.lower() for word in tweet.split() if len(word) >= 3]
    words = [stemmer.stem(w) for w in words]
    tweet_stem = ' '.join(words)
    return tweet_stem


dataset = pd.read_csv(
    '/home/kapil/PycharmProjects/Political-Opinion-Mining/csv/training.1600000.processed.noemoticon.csv',
    encoding='ISO-8859-1', header=None)
X = dataset.iloc[:, 5].values
X = pd.Series(X)
y = dataset.iloc[:, 0].values
'''
for row in range(0,1600000):
    if y[row]==4:
        y[row]=1
    else:
        y[row]=0
'''

# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)

L=[]
df = pd.DataFrame(L, columns=['string_values'])
for tweet in X_train:
    L.append(preprocess(tweet))
X_train_preprocess_tweets = pd.DataFrame(L)
for tweet in X_test:
    X_test_preprocess_tweets = preprocess(tweet)
X_train = [stem(preprocess(tweet)) for tweet in X_train]
X_test = [stem(preprocess(tweet)) for tweet in X_test]

X_train = [lemma(preprocess(tweet)) for tweet in X_train]
X_test = [lemma(preprocess(tweet)) for tweet in X_test]

vec = TfidfVectorizer(min_df=5, max_df=0.95, sublinear_tf=True, use_idf=True, ngram_range=(1, 2))
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')

svclassifier.fit(X_train_vec, y_train)

svm_predicted = svclassifier.predict(X_test_vec)

svm_score = round(svclassifier.score(X_train_vec, y_train) * 100, 2)

svm_score_test = round(svclassifier.score(X_test_vec, y_test) * 100, 2)


print('Support Vector Machine Training Score: \n', svm_score)
print('Support Vector Machine Test Score: \n', svm_score_test)
print('Coefficient: \n', svclassifier.coef_)
print('Intercept: \n', svclassifier.intercept_)
print('Accuracy: \n', metrics.accuracy_score(y_test, svm_predicted))
print('Confusion Matrix: \n', metrics.confusion_matrix(y_test, svm_predicted))
print('Classification Report: \n', metrics.classification_report(y_test, svm_predicted))




