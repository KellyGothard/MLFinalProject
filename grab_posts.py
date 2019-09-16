
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
    # Pass a list of subreddits to filter the posts that are returned
    parser.add_argument('-sr',
                        '--subreddit',
                        help='get posts from subreddit',
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

def read_months(startdate,enddate,fraction,datadir,subreddits):
    
    # Take in startdate and enddate strings and get files of dates in between
    timeframe = []
    date = datetime.strptime(startdate,'%Y-%m-%d')
    while date <= datetime.strptime(enddate,'%Y-%m-%d'):
        timeframe.append(date.strftime('%Y-%m-%d'))
        date = date + timedelta(days = 1)
    print('Reddit comments from '+str(timeframe[0])+' to '+str(timeframe[-1])+'\n') 
    
    # Read in JSON data and filter by subreddit
    posts = []
    for date in timeframe:
        lines = 0
        with gzip.open(datadir+date+'.gz','rb') as f:
            for line in f:
               lines += 1
               post = ujson.loads(line)
               if post['subreddit'] in subreddits: # Filter by subreddit here
                   posts.append([post['author'],post['subreddit'],post['created_utc'],post['body']])
    print('JSON Posts Successfully Acquired: ' + str(len(posts))+'\n')
        
    # Create DF - removed deleted authors, create datetime field from timestamp
    df = pd.DataFrame(posts,columns = ['author','subreddit','timestamp','body'])
    df = df[df['author'] != '[deleted]']
    dt = []
    for index,row in df.iterrows():
        dt.append(datetime.fromtimestamp(row['timestamp']))
    df['datetime'] = dt
    return df

def main():
    
    args = make_args()
    
    STARTDATE = args.startdate
    ENDDATE = args.enddate
    DATADIR = args.inputdir
    FRAC = args.fraction
    SR = args.subreddit
    SR = SR.split(',')
    
    print('############## BEGINNING OF RUN ##############\n')
          
    # PRAW reddit client
    reddit = praw.Reddit(client_id='gz2ObTGldvgEMg',
                         client_secret='6r1S2-WPPhyQDJmNuh8aW-b1eWY',
                         user_agent='project')
    
    # Get random subreddit
    control_subreddit = reddit.subreddit('random')
    
    SR.append(control_subreddit)
                    
    df = read_months(STARTDATE,ENDDATE,FRAC,DATADIR,SR)
    
    
    print('Number of posts: '+str(len(df))+'\n')
    print('Number of users: '+str(len(df.author.unique()))+'\n')
    print('Number of subreddits: '+str(len(df.subreddit.unique()))+'\n')
    
    print(df.groupby(['subreddit']).count())
    
    df.to_csv(STARTDATE+'_'+ENDDATE+'.csv')

if __name__=="__main__":
    main()
