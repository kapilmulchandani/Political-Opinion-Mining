from supervised.ml import preprocess_data, load_train_data, fit_tfidf, pickle_model


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = preprocess_data(load_train_data())
    vectorizer = fit_tfidf(X_train)
    pickle_model(vectorizer, 'tfidf.sav')
