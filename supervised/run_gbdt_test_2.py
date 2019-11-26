from pathlib import Path

from matplotlib import pyplot

from supervised.ml import unpickle_model, predict_test, preprocess, predict, roc
import pandas as pd
import numpy as np


def predict_test(vectorizer, model):
    fl = 'C:\\Users\\prach\\Desktop\\Political-Opinion-Mining\\Test Data Pre Processing\\test_labelled.csv'
    data = pd.read_csv(fl, encoding='utf-8-sig')
    data = data[['sentiment', 'text']].dropna()
    tweets = data['text']
    y = data['sentiment'].to_numpy()
    tweets = [preprocess(tweet) for tweet in tweets]
    X_test_vec = vectorizer.transform(tweets)
    yhat = predict(model, X_test_vec)
    yhat = np.array(yhat)
    print(y)
    print(yhat)
    print((yhat == y).mean())
    fpr, tpr, thresholds = roc(y, model.predict_proba(X_test_vec)[:, 0])
    print('fpr', fpr)
    print('tpr', tpr)
    print('thresholds', thresholds)

    # plot the roc curve for the model

    pyplot.plot(fpr, tpr, marker='.', label='Gradient Boost')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    # show the plot
    # pyplot.show()
    pyplot.savefig('gdbt_test.png')



if __name__ == '__main__':
    # fl = 'C:\\Users\\prach\\Desktop\\Political-Opinion-Mining\\Test Data Pre Processing\\test_labelled.csv'
    # data1 = pd.read_csv(fl, encoding='utf-8-sig')
    # data2 = pd.read_excel('../newTraining.xlsx')
    # i1 = set(data1['id'])
    # i2 = set(data2['id'])
    # print(len(i1), i1)
    # print(len(i2), i2)
    # print(i1 & i2)
    # t1 = data1['text']
    # t2 = data2['text']
    # vectorizer = unpickle_model('tfidf.sav')
    # x1 = vectorizer.transform(t1).toarray()
    # x2 = vectorizer.transform(t2).toarray()
    # import numpy as np
    # print(x1[0])
    # print(x2[0])
    # for jm, x in enumerate(x2):
    #     dists = [np.linalg.norm(x - y) / np.linalg.norm(x) / np.linalg.norm(y) for y in x1]
    #     im = np.argmin(dists)
    #     if dists[im] < 0.1:
    #         print(im, dists[im], t1[im], t2[jm])
    # exit()

    vectorizer = unpickle_model('tfidf.sav')
    # gbdt = unpickle_model('gbdt20000.sav')
    # gbdt = unpickle_model('gbdt5000_2.sav')
   # gbdt = unpickle_model('lr2.sav')  # 0.6523809523809524
    gbdt = unpickle_model('gbdt5000_2.sav')  # 0.6523809523809524
    predict_test(vectorizer, gbdt)
