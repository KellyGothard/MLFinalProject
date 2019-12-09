import re
import string
from collections import Counter
import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np



def clean_posts(df,stemmer = None):
    try:
        posts = list(df['body'])
    except:
        posts = list(df['comments'])
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


def get_counts_and_rank(df,stemmer = None):
    corpus = clean_posts(df,stemmer).split()
    corpus_counts = Counter(corpus)
    
    corpus_df = pd.DataFrame.from_dict(corpus_counts, orient='index').reset_index()
    corpus_df = corpus_df.rename(columns={'index':'word', 0:'count'})
    
    corpus_df["rank"] = corpus_df['count'].rank(method = 'average',ascending = False) 
    corpus_df = corpus_df.sort_values(by = ['count'],ascending = False)
    
    return corpus_df


def get_rank_divergence(c1,c2,STEM):
    corpus1_df = get_counts_and_rank(c1)
    corpus2_df = get_counts_and_rank(c2)
    
    merged = corpus1_df.merge(corpus2_df, on = 'word')
    merged['rank_div_crude'] = merged['rank_y'] - merged['rank_x']
    merged['rank_div_crude'] = (merged['rank_div_crude'] + abs(min(merged['rank_div_crude'])))/10
#    merged['rank_div'] = (1/(merged['rank_y']**(1/3))) - (1/(merged['rank_x']**(1/3)))
#    merged['importance'] = abs(merged['rank_div'] - (merged['rank_x']**2))*10

    return merged


def get_subreddit_posts(df,subreddit,banned,stemmer = None):
    df = df[df['subreddit'] == subreddit]
    posts = clean_posts(df,stemmer)
    return [subreddit, posts, banned]


def subreddit_data(path1, path2):
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
    
    rank_df = get_rank_divergence(banned_df,notbanned_df,STEM)
    
    return sr_df, rank_df


def thread_data(path, min_length = 0, max_length = 100):
    df = pd.read_csv(path)
    df = df[df['n_comments'] > min_length]
    df = df[df['n_comments'] < max_length]
    banned = df[df['banned'] == True]
    notbanned = df[df['banned'] == False]
    
    notbanned = notbanned.sample(n = len(banned))
    
    STEM = WordNetLemmatizer()
    
    threads_df = banned.append(notbanned)
    rank_df = get_rank_divergence(banned,notbanned,STEM)
    
    return threads_df, rank_df


def get_features(df, rank_df):
    # Vectorize words
    try:
        corpus = df['posts']
    except:
        corpus = df['comments']
    tfidf = TfidfVectorizer(max_features = len(df))
    
    
    # Create feature and target vector
    X = tfidf.fit_transform(corpus)
    y = df['banned']
    
    
    # Transformation
    rank_div_transform = np.tile(rank_df.rank_div_crude,(X.shape[1],1))
    X_rankdiv = (X*rank_div_transform)
    
    return X_rankdiv, X, y