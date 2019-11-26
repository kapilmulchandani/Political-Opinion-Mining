from sklearn.linear_model import LogisticRegression

from supervised.ml import preprocess_data, load_train_data_2, unpickle_model, train_lr, evaluate, pickle_model, \
    predict

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = preprocess_data(load_train_data_2())
    vectorizer = unpickle_model('tfidf.sav')
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    lr: LogisticRegression = unpickle_model('lr.sav')
    acc_train = evaluate(predict(lr, X_train_vec), y_train)
    acc_test = evaluate(predict(lr, X_test_vec), y_test)
    print(f'Train Acc = {acc_train} Test Acc = {acc_test}')
    print('-' * 50)

    lr.warm_start = True
    lr.max_iter = 20
    lr.solver = 'lbfgs'
    lr.C = 15
    lr.fit(X_train_vec, y_train)
    acc_train = evaluate(predict(lr, X_train_vec), y_train)
    acc_test = evaluate(predict(lr, X_test_vec), y_test)
    print(f'Train Acc = {acc_train} Test Acc = {acc_test}')
    print('-' * 50)
    pickle_model(lr, 'lr3.sav')

