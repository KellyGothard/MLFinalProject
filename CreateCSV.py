#!/usr/bin/env python
# coding: utf-8

# In[1]:


import bz2
import csv
import ujson
import time
from string import punctuation
import sys
from nltk.tokenize import RegexpTokenizer
import pprint


# In[2]:

in_name = sys.argv[1]
directories = in_name.split("/")
dir = '/'.join(directories[:-1])
fname = directories[-1]

REDDIT_FILE = "%s.bz2" % in_name
CSV_FILE = "CSV/%s.csv" % fname
TXT_banned = "CSV/%s_banned.txt" % fname
TXT_not_banned = "CSV/%s_notbanned.txt" %fname

NUM_POSTS_TO_READ = -1
num_read = 0

tokenizer = RegexpTokenizer(r"\w+")



BANNED_LOC = "Data/banned_reddits.txt"
banned_dict = {}
with open(BANNED_LOC) as f:
    for l in f:
        banned_dict[l.strip()[2:]] = True


f = open(CSV_FILE, 'w', newline='')
banned_file = open(TXT_banned, "w")
notbanned_file = open(TXT_not_banned, "w")

writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
writer.writerow(["id10","datetime", "subreddit", "author", "body", "tokens",  "banned", "suspect"])

def insert(post):
    writer.writerow([post["id10"], post["created_utc"], post["subreddit"], post["author"], post["body"], post["tokens"], post["banned"], post["suspect"]])
    tokens = post["tokens"].split()
    if post["banned"]:
        for token in tokens:
            banned_file.write(token + "\n")
    else:
        for token in tokens:
            notbanned_file.write(token + "\n")


# In[11]:


num_read = 0
t0 = time.time()
t1 = time.time()
f = bz2.open(REDDIT_FILE)
for line in f:
    line = line.decode("utf-8")
    num_read += 1
    if NUM_POSTS_TO_READ > 0 and num_read > NUM_POSTS_TO_READ: break
    post = ujson.loads(line)
    post["banned"] = True if post["subreddit"] in banned_dict else False
    post["suspect"] = True if post["body"] == "[removed]" or post["body"] == "[deleted]" else False
    post["id10"] = int(post["id"], 36)
    post["tokens"] = ' '.join(tokenizer.tokenize(post["body"]))
    insert(post)
    if num_read%int(1e4) == 0:
        print("Read in: %10dK posts in %.2fs (%.2fs)"%(num_read//int(1e3), time.time() - t0, time.time() - t1), flush=True)
        t1 = time.time()

f.close()

