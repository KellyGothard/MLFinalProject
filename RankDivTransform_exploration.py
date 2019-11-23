import pandas as pd
from datetime import datetime
from sklearn import metrics
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import string
from sklearn.model_selection import cross_val_predict
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import seaborn as sns
import scikitplot as skplt


def read_csv(path):
    return pd.read_csv(path)

def bow_from_df(df,stemmer = None):
    posts = list(df['body'])
    for p in posts:
        if type(p) != str:
            posts.remove(posts[posts.index(p)])
    s = ' '.join(posts)
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub(r'^https?:\/\/.*[\r\n]*', '', s, flags=re.MULTILINE)
    document = s.lower()
    if stemmer:
        document = document.split()
        document = [stemmer.lemmatize(word) for word in document]
        ' '.join(document)
    return document


def word_counts(df,stemmer = None):
    corpus = bow_from_df(df,stemmer).split()
    corpus_counts = Counter(corpus)
    
    corpus_df = pd.DataFrame.from_dict(corpus_counts, orient='index').reset_index()
    corpus_df = corpus_df.rename(columns={'index':'word', 0:'count'})
    
    corpus_df["rank"] = corpus_df['count'].rank(method = 'average',ascending = False) 
    corpus_df = corpus_df.sort_values(by = ['count'],ascending = False)
    
    return corpus_df


def rankdiv_scatter(rank_df,col):
    rank_df['log_rank_div'] = np.log10(rank_df[col] + 1)
    rank_df['log_rank_x'] = np.log10(rank_df['rank_x'] + 1)
    
    fig, ax = plt.subplots(figsize=(12,16))
    ax.scatter(x = rank_df['log_rank_div'] , y = rank_df['log_rank_x'], s = 1)
    
    for i, txt in enumerate(rank_df['word']):
        if random.uniform(0,1) < 0.05:
            if col == 'rank_div':
                ax.annotate(txt, (rank_df['log_rank_div'].iloc[i], rank_df['log_rank_x'].iloc[i]),size = 1000*abs(rank_df['log_rank_div'].iloc[i]), alpha = 0.5)
            elif col == 'rank_div_crude':
                ax.annotate(txt, (rank_df['log_rank_div'].iloc[i], rank_df['log_rank_x'].iloc[i]),size = 4*rank_df['log_rank_div'].iloc[i], alpha = 0.8)
            
    plt.xlabel('rank_div',fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('rank',fontsize=16)
    plt.gca().invert_yaxis()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
#    plt.savefig('test_rankdiv_scatter.png')
    plt.show()
    plt.close()


def rank_divergence(c1,c2,STEM):
    corpus1_df = word_counts(c1)
    corpus2_df = word_counts(c2)
    
    merged = corpus1_df.merge(corpus2_df, on = 'word')
    merged['rank_div_crude'] = abs(merged['rank_y'] - merged['rank_x'])
    merged['rank_div'] = (1/(merged['rank_y']**(1/3))) - (1/(merged['rank_x']**(1/3)))
    merged['importance'] = abs(merged['rank_div'] - (merged['rank_x']**2))*10

    return merged


def plots(df):
    plt.hist(df['rank_div_crude'])
    plt.show()
    plt.close()
    
    plt.hist(np.log10(df['rank_div_crude'] + 1))
    plt.show()
    plt.close()
    
    plt.hist(df['rank_div_crude'])
    plt.show()
    plt.close()
    
    plt.hist(df['importance'])
    plt.show()
    plt.close()
    
    rankdiv_scatter(df,'rank_div')
    

def get_subreddit_posts(df,subreddit,banned,stemmer = None):
    df = df[df['subreddit'] == subreddit]
    posts = bow_from_df(df,stemmer)
    return [subreddit, posts, banned]


def confusion_plot(array,name):
    
    df_cm = pd.DataFrame(array, index = [i for i in "AB"],
                      columns = [i for i in "AB"])
    plt.figure(figsize = (8,8))
    plt.title(name)    
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    sns.set(font_scale=1.4)#for label size
    sns.heatmap(df_cm, annot=True, vmin = 0, vmax = 117, cmap = 'viridis', annot_kws={"size": 16})# font size

def roc(clf, X,y):
    y_probas = clf.predict_proba(X)
    skplt.metrics.plot_roc(y, y_probas)

def cross_val(clf,X,y,name):
    y_pred = cross_val_predict(clf, X, y, cv=10)
#    print(metrics.classification_report(y, y_pred))
    conf = np.array(metrics.confusion_matrix(y, y_pred))
    print(name)
    print(conf)
    #confusion_plot(conf, name)
#    skplt.metrics.plot_confusion_matrix(y, y_pred, figsize=(12,12))
#    roc(clf,X,y)
    return metrics.f1_score(y,y_pred,pos_label=1, average='binary')


def rankdiv_vs_not(clf,X,y,name):
    clf = clf.fit(X, y)
    return cross_val(clf, X, y, name)


def word_embedding_hist(X):
    plt.hist(X)
    plt.show()
    
    
def get_data(path1, path2):
    banned_df = pd.read_csv(path1)
    notbanned_df = pd.read_csv(path2)
    temp = notbanned_df.groupby('subreddit').count()
    temp = temp.reset_index()
    
    sample_srs = list(temp['subreddit'].sample(n=117))
    notbanned_df = notbanned_df[notbanned_df['subreddit'].isin(sample_srs)]
    
    data = []
    STEM = WordNetLemmatizer()
    
    for subreddit in banned_df['subreddit'].unique():
        data.append(get_subreddit_posts(banned_df,subreddit,1))
    
    for subreddit in notbanned_df['subreddit'].unique():
        data.append(get_subreddit_posts(notbanned_df,subreddit,0)) 
    
    sr_df = pd.DataFrame(data, columns = ['subreddit','posts','banned'])
    rank_df = rank_divergence(banned_df,notbanned_df,STEM)
    
    return sr_df, rank_df
        

# Get data
path1 = 'banned.csv'
path2 = 'notbanned.csv'
df, rank_df = get_data(path1, path2)

# Vectorize words
corpus = df['posts']
tfidf = CountVectorizer(max_features = len(df))


# Create feature and target vector
X = tfidf.fit_transform(corpus)
y = df['banned']


# Transformation
rank_div_transform = np.tile(rank_df.rank_div,(X.shape[1],1))
X_rankdiv = (X*rank_div_transform)

importance_transform = np.tile(rank_df.importance,(X.shape[1],1))
X_rankdiv = (X*importance_transform)
    
scores = []
clf = MultinomialNB()
name = 'MNB: TF-IDF'
d = rankdiv_vs_not(clf,X,y,name)
scores.append(['MNB','TF-IDF',d])
name = 'MNB: TF-IDF+RD'
d = rankdiv_vs_not(clf,X_rankdiv,y,name)
scores.append(['MNB','TF-IDF+RD',d])


clf = SVC()
name = 'SVC: TF-IDF'
d = rankdiv_vs_not(clf,X,y,name)
scores.append(['SVC','TF-IDF',d])
name = 'SVC: TF-IDF+RD'
d = rankdiv_vs_not(clf,X_rankdiv,y,name)
scores.append(['SVC','TF-IDF+RD',d])
    

df = pd.DataFrame(scores,columns=['clf','vec','f1'])
plt.figure(figsize=(10, 6))
sns.barplot(x="clf", hue="vec", y="f1", data=df)
plt.legend(bbox_to_anchor=(1, 1), loc=9, borderaxespad=0.4)
plt.xlabel('Classifier')
plt.ylabel('Banned: F1 Score')
plt.savefig('clf_barplot.png')