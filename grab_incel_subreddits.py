
############################# Imports #############################
from datetime import timedelta
import ujson
import pandas as pd
from datetime import datetime
import praw
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.svm import SVC
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import argparse
import gzip
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def make_args():
    description = 'Generalized jobs submitter for PBS on VACC. Tailored to jobs that can be chunked based on datetime.' \
                  ' Scripts to be run MUST have -o output argument. \n Output will be saved in log files with the first 3' \
                  ' characters of args.flexargs and the start date for the job'
    parser = argparse.ArgumentParser(description=description,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i',
                        '--inputdir',
                        help='input directory',
                        required=True,
                        type=str)
    parser.add_argument('-o',
                        '--outdir',
                        help='output directory (will be passed to args.script with -o argument)',
                        required=True,
                        type=str)
    parser.add_argument('--checkcompleted',
                        help='where to check for completed days.',
                        required=False,
                        type=str,
                        default=None)
    parser.add_argument('-s',
                        '--startdate',
                        help='optional date to constrain the run',
                        required=False,
                        default=None,
                        type=str)
    parser.add_argument('-e',
                        '--enddate',
                        help='optional date to constrain the run',
                        required=False,
                        default=None,
                        type=str)
    parser.add_argument('-d',
                        '--datadir',
                        help='directory that data lives in',
                        required=False,
                        default=None,
                        type=str)
    parser.add_argument('-f',
                        '--fraction',
                        help='use fraction of posts',
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

def read_months(startdate,enddate,fraction,datadir):
    timeframe = []
    date = datetime.strptime(startdate,'%Y-%m-%d')
    while date <= datetime.strptime(enddate,'%Y-%m-%d'):
        timeframe.append(date.strftime('%Y-%m-%d'))
        date = date + timedelta(days = 1)
    print('Reddit comments from '+str(timeframe[0])+' to '+str(timeframe[-1])+'\n')    
    # Read in JSON data
    posts = []
    for date in timeframe:
        lines = 0
        with gzip.open(datadir+date+'.gz','rb') as f:
            for line in f:
               lines += 1
               post = ujson.loads(line)
               posts.append([post['author'],post['subreddit'],post['created_utc'],post['body']])
    print('JSON Posts Successfully Acquired: ' + str(len(posts))+'\n')
        
    # Create DF
    df = pd.DataFrame(posts,columns = ['author','subreddit','timestamp','body'])
    df = df[df['author'] != '[deleted]']
    dt = []
    for index,row in df.iterrows():
        dt.append(datetime.fromtimestamp(row['timestamp']))
    df['datetime'] = dt
    return df

###################### Subreddit Classification ######################

def bow_from_df(df,subreddit,stemmer):
    punctuations = '''!()\-[]{};:'"\,<>./?@#$%^&*_~|''';
    posts = list(df[df['subreddit'] == subreddit]['body'])
    s = ''
    for post in posts:
        for char in punctuations:
            post = post.replace(char, '')
        posttext = post.replace('\n','') + ' '
        s += posttext
    document = re.sub(r'^https?:\/\/.*[\r\n]*', '', s, flags=re.MULTILINE)
    document = document.lower()
    document = document.split()
    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    for stopword in set(stopwords.words('english')):
        document.replace(stopword,'')
    document = document.split()
    return document

def compare_corpora(df,bow):
    stemmer = PorterStemmer()
    # PRAW get random subreddits
    reddit = praw.Reddit(client_id='gz2ObTGldvgEMg',
                         client_secret='6r1S2-WPPhyQDJmNuh8aW-b1eWY',
                         user_agent='project')
    control_subreddit = reddit.subreddit('random')
    control_bow = bow_from_df(df,control_subreddit,stemmer)
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfs = tfidf.fit_transform([bow,control_bow])
    return tfs
    

def main():
    
    args = make_args()
    
    STARTDATE = args.startdate
    ENDDATE = args.enddate
    DATADIR = args.inputdir
    FRAC = args.fraction
    STEM = WordNetLemmatizer()
    
    print('############## BEGINNING OF RUN ##############\n')
                    
    df = read_months(STARTDATE,ENDDATE,FRAC,DATADIR)
    bow = bow_from_df(df,'Incels',STEM)
    tfs = compare_corpora(df,bow)
    
    print(tfs)
    
    print('Number of posts: '+str(len(df))+'\n')
    print('Number of users: '+str(len(df.author.unique()))+'\n')
    print('Number of subreddits: '+str(len(df.subreddit.unique()))+'\n')
    

if __name__=="__main__":
    main()
