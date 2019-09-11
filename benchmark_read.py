#### IMPORTS #####
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
import time

def human_readable(num):
    if num < 1e3:
        return str(num)
    elif num < 1e6:
        return "%dK"%(num//1e3)
    elif num < 1e9:
        return "%dM"%(num//1e6)
    elif num < 1e12:
        return "%dG"%(num//1e9)
    elif num < 1e15:
        return "%dT"%(num//1e12)
    elif num < 1e18:
        return "%dP"%(num//1e15)
    else:
        return num

# Read in JSON data
NUM_POSTS_TO_READ = -1
MSG_PREV = 30

MESSAGES_LOC = "Data/RC_2019-01"

header = ("%-23s | %-20s | %"+str(MSG_PREV+10)+"s")%("Subreddit", "User", "Message")
print(header)
print("-"*len(header))

t0 = time.time() 
num_posts_read_at_t0 = 0

with open(MESSAGES_LOC, "r") as f:
    num_read = 0
    for line in f:
        num_read += 1
        if NUM_POSTS_TO_READ != -1 and num_read > NUM_POSTS_TO_READ: break
        post = ujson.loads(line)
        #print(("/r/%-20s | %-20s | %"+str(MSG_PREV+10)+"s")%( post['subreddit'], post['author'], repr(post['body'][:MSG_PREV]))) #post['created_utc']
        if time.time() > t0 + 0.5:
            t1 = time.time()
            read_speed = (num_read - num_posts_read_at_t0)/(t1-t0)
            print("Read in %s [%s/s] posts"%(human_readable(num_read), human_readable(read_speed)), flush=True)
            t0 = t1
            num_posts_read_at_t0 = num_read


