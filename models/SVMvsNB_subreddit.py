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
             

def main():

    # Get data
    path1 = '../data/banned.csv'
    path2 = '../data/notbanned.csv'
    
    df, rank_df = proc.subreddit_data(path1, path2)
    df = df.rename(columns={"posts": "comments"})
#    plots.rankdiv_scatter(rank_df,'rank_div_crude', 'Subreddits')
#    plots.rankdiv_shift(rank_df,'Subreddits')
#    plots.rank_divergence_dist_plot(rank_df, 'Subreddits')
#    
#    X_rankdiv, X, y = proc.get_features(df, rank_df)
#    
#    
#    scores = []
#    clf = MultinomialNB()
#    name = 'MNB: TF-IDF'
#    d = fit(clf,X,y,name)
#    scores.append(['MNB','TF-IDF',d])
#    name = 'MNB: TF-IDF+RD'
#    d = fit(clf,X_rankdiv,y,name)
#    scores.append(['MNB','TF-IDF+RD',d])
#    
#    
#
#    clf = SVC(gamma='auto')
#    clf.probability = True
#    name = 'SVC: TF-IDF'
#    d = fit(clf,X,y,name)
#    scores.append(['SVC','TF-IDF',d])
#    name = 'SVC: TF-IDF+RD'
#    d = fit(clf,X_rankdiv,y,name)
#    scores.append(['SVC','TF-IDF+RD',d])
#    
#    
#    
#    data = pd.DataFrame(scores,columns=['clf','vec','f1'])
#    plt.figure(figsize=(10, 6))
#    sns.barplot(x="clf", hue="vec", y="f1", data=data)
#    plt.legend(bbox_to_anchor=(1, 1), loc=9, borderaxespad=0.4)
#    plt.xlabel('Classifier')
#    plt.ylabel('Banned: F1 Score')
#    plt.savefig('../images/subreddits_clf_barplot.png')
    
    
    plots.words_per_row(df,'Subreddit')

        
    
if __name__=="__main__":
    main()
