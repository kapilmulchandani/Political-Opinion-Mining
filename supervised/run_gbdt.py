from pathlib import Path

from supervised.ml import unpickle_model, predict_test

if __name__ == '__main__':
    vectorizer = unpickle_model('tfidf.sav')
    lr = unpickle_model('gbdt.sav')
    predict_test(vectorizer, lr, Path('gbdt_test_results'))
