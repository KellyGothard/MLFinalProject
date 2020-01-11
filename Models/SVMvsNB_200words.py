import pandas as pd
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import seaborn as sns
import scikitplot as skplt

import plots
import data_processing as proc


def cross_val(clf,X,y,name):
    print(name)
    y_pred = cross_val_predict(clf, X, y, cv=10)
    print(metrics.classification_report(y, y_pred))
    conf = np.array(metrics.confusion_matrix(y, y_pred))
    print(conf)
    y_probas = clf.predict_proba(X)
    skplt.metrics.plot_roc_curve(y, y_probas, title=name+' ROC Curves', curves='each_class')
    return metrics.f1_score(y,y_pred,pos_label=1, average='binary')


def fit(clf,X,y,name):
    clf = clf.fit(X, y)
    return cross_val(clf, X, y, name)

def predict(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    
    print(metrics.classification_report(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))
             

def main():

    # Get data
    path1 = '../Data/200_words_10M_banned.csv'
    path2 = '../Data/200_words_10M_notbanned.csv'
    pathtest = '..Ddata/200_words_10M_test.csv'
    
    df, rank_df = proc.word200_data(path1, path2)
    
    X_rankdiv, X, y = proc.get_features(df, rank_df)
    
#    print(X.shape)
#    print(y.shape)
#    
#    testdf, testrank_df = proc.subreddit_data_singlefile(pathtest)
#    Xtest_rankdiv, X_test, y_test = proc.get_features(testdf, testrank_df)    
    
#    print(Xtest_rankdiv.shape)
    
    scores = []
    clf = MultinomialNB()
    name = 'MNB: TF-IDF'
    d = fit(clf,X,y,name)
    scores.append(['MNB','TF-IDF',d])
    name = 'MNB: TF-IDF+RD'
    d = fit(clf,X_rankdiv,y,name)
    scores.append(['MNB','TF-IDF+RD',d])
    
    clf = SVC(gamma='auto')
    clf.probability = True
    name = 'SVC: TF-IDF'
    d = fit(clf,X,y,name)
    scores.append(['SVC','TF-IDF',d])
    name = 'SVC: TF-IDF+RD'
    d = fit(clf,X_rankdiv,y,name)
    scores.append(['SVC','TF-IDF+RD',d])
    
    
#    clf = MultinomialNB()
#    name = 'MNB: TF-IDF'
#    print(name)
#    clf = clf.fit(X, y)
#    predict(clf, X, y_test)
#    name = 'MNB: TF-IDF+RD'
#    print(name)
#    clf = clf.fit(X_rankdiv, y)
#    predict(clf, X_rankdiv, y_test)
#    
#    clf = SVC(gamma='auto')
#    clf.probability = True
#    name = 'SVC: TF-IDF'
#    print(name)
#    clf = clf.fit(X, y)
#    predict(clf, X, y_test)
#    name = 'SVC: TF-IDF+RD'
#    print(name)
#    clf = clf.fit(X_rankdiv, y)
#    predict(clf, X_rankdiv, y_test)
#    
    
    data = pd.DataFrame(scores,columns=['clf','vec','f1'])
    plt.figure(figsize=(10, 6))
    sns.barplot(x="clf", hue="vec", y="f1", data=data)
    plt.legend(bbox_to_anchor=(1, 1), loc=9, borderaxespad=0.4)
    plt.xlabel('Classifier')
    plt.ylabel('Banned: F1 Score')
    plt.savefig('../images/words200_clf_barplot.png')
#    
    
#    plots.words_per_row(df,'Subreddit')

        
    
if __name__=="__main__":
    main()
