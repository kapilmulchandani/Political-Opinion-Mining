from matplotlib import pyplot

from supervised.ml import preprocess_data, load_train_data_2, unpickle_model, train_lr, evaluate, pickle_model, \
    predict, confusion_matrix, roc

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = preprocess_data(load_train_data_2())
    vectorizer = unpickle_model('tfidf.sav')
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    lr = train_lr(X_train_vec, y_train, C=10)
    acc_train = evaluate(predict(lr, X_train_vec), y_train)
    acc_test = evaluate(predict(lr, X_test_vec), y_test)
    conf_mat = confusion_matrix(predict(lr, X_test_vec), y_test)
    # fpr, tpr, thresholds = roc(y_test, lr.predict_proba(X_test_vec)[:, 0])
    # print('fpr', fpr)
    # print('tpr', tpr)
    # print('thresholds', thresholds)
    #
    # # plot the roc curve for the model
    #
    # pyplot.plot(fpr, tpr, marker='.', label='Logistic')
    # # axis labels
    # pyplot.xlabel('False Positive Rate')
    # pyplot.ylabel('True Positive Rate')
    # # show the legend
    # pyplot.legend()
    # # show the plot
    # pyplot.show()
    # print(f'Train Acc = {acc_train} Test Acc = {acc_test}')
    # print(f'Confusion Matrix = {conf_mat}')
    # # Train Acc = 0.853759375 Test Acc = 0.824384375
    # pickle_model(lr, 'lr2.sav')
