from collections import Counter
import pandas as pd
import re
import string


def bow_from_df(df,stemmer = None):
    '''
    Takes in a dataframe of reddit posts where the text column is called 'body'
    '''
    
    posts = list(df['body'])
    
    for p in posts:
        if type(p) != str:
            posts.remove(posts[posts.index(p)])
    # Text cleaning: remove punctuation, URLs, lowering
    s = ' '.join(posts)
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub(r'^https?:\/\/.*[\r\n]*', '', s, flags=re.MULTILINE)
    document = s.lower()
    
    # Stemming is optional
    if stemmer:
        document = document.split()
        document = [stemmer.lemmatize(word) for word in document]
        ' '.join(document)
        
    # Words separated by spaces
    return document


def word_counts(df,stemmer = None):
    corpus = bow_from_df(df,stemmer).split()
    corpus_counts = Counter(corpus)
    
    corpus_df = pd.DataFrame.from_dict(corpus_counts, orient='index').reset_index()
    corpus_df = corpus_df.rename(columns={'index':'word', 0:'count'})
    
    corpus_df["rank"] = corpus_df['count'].rank(method = 'average',ascending = False) 
    corpus_df = corpus_df.sort_values(by = ['count'],ascending = False)
    
    return corpus_df


def rank_divergence(c1,c2,STEM):
    '''
    Takes in two corpora, returns a dataframe 
    with each row containing a word and its rank divergence
    '''
    
    corpus1_df = word_counts(c1)
    corpus2_df = word_counts(c2)
    
    merged = corpus1_df.merge(corpus2_df, on = 'word')
    merged['rank_div'] = (1/(merged['rank_y']**(1/3))) - (1/(merged['rank_x']**(1/3)))

    return merged