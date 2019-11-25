from supervised.ml import preprocess_data, load_train_data, unpickle_model, train_lr, evaluate, pickle_model, \
    predict

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = preprocess_data(load_train_data())
    vectorizer = unpickle_model('tfidf.sav')
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    lr = train_lr(X_train_vec, y_train)
    acc_train = evaluate(predict(lr, X_train_vec), y_train)
    acc_test = evaluate(predict(lr, X_test_vec), y_test)
    print(f'Train Acc = {acc_train} Test Acc = {acc_test}')
    # Train Acc = 0.853759375 Test Acc = 0.824384375
    pickle_model(lr, 'lr.sav')
