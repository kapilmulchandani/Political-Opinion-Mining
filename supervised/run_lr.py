from pathlib import Path

from supervised.ml import unpickle_model, predict_test

if __name__ == '__main__':
    vectorizer = unpickle_model('tfidf.sav')
    # lr = unpickle_model('lr.sav')
    lr = unpickle_model('lr2.sav')
    predict_test(vectorizer, lr, Path('logistic_test_results'))

