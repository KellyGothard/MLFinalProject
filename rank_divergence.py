
############################# Imports #############################
from datetime import timedelta
import ujson
import pandas as pd
from datetime import datetime
import praw
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import argparse
import gzip
from nltk.stem.porter import PorterStemmer
from collections import Counter
import string
import matplotlib.pyplot as plt

def make_args():
    description = 'Generalized jobs submitter for PBS on VACC. Tailored to jobs that can be chunked based on datetime.' \
                  ' Scripts to be run MUST have -o output argument. \n Output will be saved in log files with the first 3' \
                  ' characters of args.flexargs and the start date for the job'
    # Specify directory that reddit posts live in in .pbs script
    parser = argparse.ArgumentParser(description=description,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i1',
                        '--inputdir1',
                        help='input directory',
                        required=True,
                        type=str)
    parser.add_argument('-i2',
                        '--inputdir2',
                        help='input directory',
                        required=True,
                        type=str)
    # Where your output should be dumped - recommend making a folder for each pbs script 
    # output to keep it clean, as well as a timestamp so you know the order of error logs
    parser.add_argument('-o',
                        '--outdir',
                        help='output directory (will be passed to args.script with -o argument)',
                        required=True,
                        type=str)
    # Get posts starting on this date
    parser.add_argument('-s',
                        '--startdate',
                        help='optional date to constrain the run',
                        required=False,
                        default=None,
                        type=str)
    # Get posts ending on this date (use same date as startdate if you only want one day of posts)
    parser.add_argument('-e',
                        '--enddate',
                        help='optional date to constrain the run',
                        required=False,
                        default=None,
                        type=str)
    # Take a sample of the posts (float as a percentage of the lines read in) - useful for testing
    parser.add_argument('-f',
                        '--fraction',
                        help='use fraction of posts',
                        required=False,
                        default=None,
                        type=str)
    # Control subreddit for comparision - should already be in csv
    parser.add_argument('-c',
                        '--control',
                        help='Control subreddit for comparision - should already be in csv',
                        required=False,
                        default=None,
                        type=str)   
    return parser.parse_args()
    
def valid_date(d):
    try:
        return datetime.strptime(d, "%Y-%m-%d")
    except ValueError:
        msg = "Invalid date format in provided input: '{}'.".format(d)
        raise argparse.ArgumentTypeError(msg)

########################## Read in Data ##########################
        
def read_csv(path):
    return pd.read_csv(path)

###################### Subreddit Classification ######################

def bow_from_df(df,stemmer):
    
    # Get posts as a list
    posts = list(df['body'])
    for p in posts:
        if type(p) != str:
            print(p)
    s = ' '.join(posts)
    s = s.translate(str.maketrans('', '', string.punctuation))

    # Lower case, split string of words into list of words, lemmatize
    document = s.lower()
    document = document.split()
    #document = [stemmer.lemmatize(word) for word in document]
    print(document[200:400])
    return document

def word_counts(df,stemmer):
    
    # Bag of words
    corpus = bow_from_df(df,stemmer)
    
    # Get counts for each word in bag
    corpus_counts = Counter(corpus)
    # Create dataframe of words and counts
    corpus_df = pd.DataFrame.from_dict(corpus_counts, orient='index').reset_index()
    corpus_df = corpus_df.rename(columns={'index':'word', 0:'count'})
    # Create column for rank in dataframe
    corpus_df["rank"] = corpus_df['count'].rank(method = 'average',ascending = False) 
    corpus_df = corpus_df.sort_values(by = ['count'],ascending = False)
    
    return corpus_df

def rank_divergence(corpus1_df,corpus2_df):
    merged = corpus1_df.merge(corpus2_df, on = 'word')
    return merged

def main():
    args = make_args()
    
    path1 = args.inputdir1
    path2 = args.inputdir2
    STEM = WordNetLemmatizer()
    
    print('############## BEGINNING OF RUN ##############\n')
          
    df1 = read_csv(path1)
    df2 = read_csv(path2)
    
    corpus1_df = word_counts(df1,STEM)
    print(corpus1_df.iloc[200:500])
    corpus2_df = word_counts(df2,STEM)
    
    rank_df = rank_divergence(corpus1_df,corpus2_df)
    
    sr1 = path1.split('_')[0]
    sr2 = path2.split('_')[0]
    
    rank_df['rank_div'] = rank_df['rank_y'] - rank_df['rank_x']
    rank_df = rank_df.sort_values(['rank_y','rank_div'])
    
    plt.hist(rank_df['rank_div'])
    plt.title('Rank Div Dist: r/Braincel and r/WaltDisneyWorld Words')
    plt.savefig(sr1+'_'+sr2+'_rankdivhist.png')
    plt.close()
    
    plt.scatter(rank_df['rank_x'],rank_df['rank_y'])
    plt.title('Word Rank in r/Braincel vs Rank in r/WaltDisneyWorld')
    plt.savefig(sr1+'_'+sr2+'_rankdivscatter.png')
    plt.close()

    plt.loglog(rank_df['rank_x'],rank_df['count_x'])
    plt.title('Zipf Dist of r/Braincel Words')
    plt.savefig(sr1+'_'+sr2+'_zipfs1.png')
    plt.close()

    plt.loglog(rank_df['rank_y'],rank_df['count_y'])
    plt.title('Zipf Dist of r/WaltDisneyWorld Words')
    plt.savefig(sr1+'_'+sr2+'_zipfs2.png')
    plt.close()

    rank_df.to_csv(sr1+'_'+sr2+'_rankdiv.csv')
    
    corpus1_df.sort_values('rank')
    corpus1_df.to_csv(sr1+'_rankdiv.csv')
    
if __name__=="__main__":
    main()
