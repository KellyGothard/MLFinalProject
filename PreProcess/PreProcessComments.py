# Converts the Reddit comments from a given month into 2 files that contain banned and unbanned posts.
# This is a neded pre-processing step for generating training data.

# this program reads in each comment and extracts the words in order from it (removing punctuation et al.)
# If the comment was made in a now banned subreddit then it's words are added to the banned file.
# otherwise they are added to the not banned file.


import bz2
import csv
import ujson
import time
from string import punctuation
import sys
from nltk.tokenize import RegexpTokenizer
import pprint


# Determine location of file to process.
in_name = "RC_2016-10"

REDDIT_FILE = "../Data/%s.bz2" % in_name

TXT_banned = "../Data/Generated/%s_banned.txt" % in_name
TXT_not_banned = "../Data/Generated/%s_notbanned.txt" %in_name

# construct a tokenizer for cleaning posts.
tokenizer = RegexpTokenizer(r"\w+")

# load list of banned subreddits.
BANNED_LOC = "../Data/banned_reddits.txt"
banned_dict = {}
with open(BANNED_LOC) as f:
    for l in f:
        banned_dict[l.strip()[2:]] = True


# create files.
banned_file = open(TXT_banned, "w")
notbanned_file = open(TXT_not_banned, "w")

# method for updating training data files.
def insert(tokens, banned):
    if banned:
        for token in tokens:
            banned_file.write(token + "\n")
    else:
        for token in tokens:
            notbanned_file.write(token + "\n")



# begin processing all posts in the input file.
num_read = 0
t0 = time.time()
t1 = t0
f = bz2.open(REDDIT_FILE)
for line in f:
    line = line.decode("utf-8")
    num_read += 1
    post = ujson.loads(line)
    banned =  True if post["subreddit"] in banned_dict else False
    tokens = tokenizer.tokenize(post["body"]) 
    insert(tokens, banned)

    if num_read%int(1e4) == 0:
        print("Read in: %10dK posts in %.2fs (%.2fs)"%(num_read//int(1e3), time.time() - t0, time.time() - t1), flush=True)
        t1 = time.time()

f.close()

