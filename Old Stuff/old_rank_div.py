
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

def make_args():
    description = 'Generalized jobs submitter for PBS on VACC. Tailored to jobs that can be chunked based on datetime.' \
                  ' Scripts to be run MUST have -o output argument. \n Output will be saved in log files with the first 3' \
                  ' characters of args.flexargs and the start date for the job'
    # Specify directory that reddit posts live in in .pbs script
    parser = argparse.ArgumentParser(description=description,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i',
                        '--inputdir',
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
    punctuations = '''!()\-[]{};:'"\,<>./?@#$%^&*_~|''';
    s = ''
    
    # Get posts as a list
    posts = list(df['body'])
    
    # Remove punctuation and stopwords, create large string of posts, s
    for post in posts:
        print(post)
        for char in punctuations:
            post = post.replace(char, '')
            for stopword in set(stopwords.words('english')):
                post = post.replace(stopword,'')
        posttext = post.replace('\n','') + ' '
        s += posttext
    
    # Remove URLs
    document = re.sub(r'^https?:\/\/.*[\r\n]*', '', s, flags=re.MULTILINE)
    
    # Lower case, split string of words into list of words, lemmatize
    document = document.lower()
    document = document.split()
    document = [stemmer.lemmatize(word) for word in document]
    
    return document

def word_counts(df,subreddit):
    
    # Bag of words
    corpus = bow_from_df(df[df['subreddit'] == subreddit])
    
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
    merged = pd.merge([corpus1_df,corpus2_df])

def main():
    
    args = make_args()
    
    path = args.inputdir
    STEM = WordNetLemmatizer()
    CONTROL = args.control
    
    print('############## BEGINNING OF RUN ##############\n')
          
    df = read_csv(path)
    
    corpus1_df = word_counts(df,subreddit1)
    corpus2_df = word_counts(df,subreddit2)
    
    

if __name__=="__main__":
    main()
