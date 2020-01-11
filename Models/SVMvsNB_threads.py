import pandas as pd
from sklearn import metrics
import re
from collections import Counter
import string
from sklearn.model_selection import cross_val_predict
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
import scikitplot as skplt

import plots
import data_processing as proc

import warnings
warnings.filterwarnings('ignore')


#def svc_param_selection(X, y, nfolds):
#    Cs = [0.001, 0.01, 0.1, 1, 10]
#    gammas = [0.001, 0.01, 0.1, 1]
#    param_grid = {'C': Cs, 'gamma' : gammas}
#    gs = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds)
#    gs.fit(X, y)
#    return gs
#        



def cross_val(clf,X,y,name):
#    print(name)
    y_pred = cross_val_predict(clf, X, y, cv=10)
#    print(metrics.classification_report(y, y_pred))
#    conf = np.array(metrics.confusion_matrix(y, y_pred))
#    print(conf)
#    y_probas = clf.predict_proba(X)
#    skplt.metrics.plot_roc_curve(y, y_probas, title=name+' ROC Curves', curves='each_class')
    return metrics.f1_score(y,y_pred,pos_label=1, average='binary')


def fit(clf,X,y,name):
    clf = clf.fit(X, y)
    return cross_val(clf, X, y, name)
             

def main():    
    
    path = '../data/reddit_threads.csv'
    
#    scores = []
#    for i in range(250):
        
    df, rank_df = proc.thread_data(path, 1)
    
#        plots.rankdiv_scatter(rank_df,'rank_div_crude', 'Threads')
#        plots.rankdiv_shift(rank_df,'Threads')
#        plots.rank_divergence_dist_plot(rank_df, 'Threads')
#        
#        X_rankdiv, X, y = proc.get_features(df, rank_df)
#    
#        clf = MultinomialNB()
#        name = 'MNB: TF-IDF'
#        d = fit(clf,X,y,name)
#        scores.append(['MNB','TF-IDF',d])
#        name = 'MNB: TF-IDF+RD'
#        d = fit(clf,X_rankdiv,y,name)
#        scores.append(['MNB','TF-IDF+RD',d])
#        
#        
#    
#        clf = SVC(gamma='auto')
#        clf.probability = True
#        name = 'SVC: TF-IDF'
#        d = fit(clf,X,y,name)
#        scores.append(['SVC','TF-IDF',d])
#        name = 'SVC: TF-IDF+RD'
#        d = fit(clf,X_rankdiv,y,name)
#        scores.append(['SVC','TF-IDF+RD',d])
#    
#    
#    data = pd.DataFrame(scores,columns=['clf','vec','f1'])
#    
#    for clf in data.clf.unique():
#        for transform in data.vec.unique():
#            temp = data[data['clf'] == clf]
#            temp = temp[temp['vec'] == transform]
#            plt.hist(temp['f1'], bins = len(temp))
#            plt.title(clf+' '+transform+' Bannable F1 Distribution')
#            plt.show()
#            
#    plt.figure(figsize=(10, 6))
#    sns.barplot(x="clf", hue="vec", y="f1", data=df)
#    plt.legend(bbox_to_anchor=(1, 1), loc=9, borderaxespad=0.4)
#    plt.xlabel('Classifier')
#    plt.ylabel('Banned: F1 Score')
#    plt.savefig('../images/threads_clf_barplot.png')
    
    plots.words_per_row(df,'Thread')

if __name__=="__main__":
    main()
