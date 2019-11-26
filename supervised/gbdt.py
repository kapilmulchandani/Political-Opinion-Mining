from supervised.ml import preprocess_data, load_train_data, unpickle_model, evaluate, pickle_model, \
    train_gbdt, predict

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = preprocess_data(load_train_data())
    vectorizer = unpickle_model('tfidf.sav')
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = train_gbdt(X_train_vec, y_train, n_est=20000)
    acc_train = evaluate(predict(model, X_train_vec), y_train)
    acc_test = evaluate(predict(model, X_test_vec), y_test)
    print(f'Train Acc = {acc_train} Test Acc = {acc_test}')
    # n_est = 100: Train Acc = 0.5661953125 Test Acc = 0.56625625
    # n_est = 1000: Train Acc = 0.71187109375 Test Acc = 0.708015625
    pickle_model(model, 'gbdt20000.sav')

    model = train_gbdt(X_train_vec, y_train, n_est=30000)
    acc_train = evaluate(predict(model, X_train_vec), y_train)
    acc_test = evaluate(predict(model, X_test_vec), y_test)
    print(f'Train Acc = {acc_train} Test Acc = {acc_test}')
    # n_est = 100: Train Acc = 0.5661953125 Test Acc = 0.56625625
    # n_est = 1000: Train Acc = 0.71187109375 Test Acc = 0.708015625
    # n_est = 5000: Train Acc = 0.78938203125 Test Acc = 0.7805875
    # n_est = 10000 Train Acc = 0.8103390625 Test Acc = 0.797578125
    # n_est = 15000 Train Acc = 0.82
    # n_est = 20000 Train Acc = 0.829825 Test Acc = 0.810390625
    pickle_model(model, 'gbdt30000.sav')
