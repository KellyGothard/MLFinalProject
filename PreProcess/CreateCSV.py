#!/usr/bin/env python
# coding: utf-8

# In[1]:


import bz2
import csv
import ujson
import time
from string import punctuation
import sys

import pprint


# In[2]:

in_name = sys.argv[1]
directories = in_name.split("/")
dir = '/'.join(directories[:-1])
fname = directories[-1]

REDDIT_FILE = "%s.bz2" % in_name
CSV_FILE = "CSV/%s.csv" % fname

NUM_POSTS_TO_READ = -1
num_read = 0


def clean_text(text):
    exclude = set(punctuation) # Keep a set of "bad" characters.
    list_letters_noPunct = [ char for char in text if char not in exclude ]
    
    # Now we have a list of LETTERS, *join* them back together to get words:
    text_noPunct =  "".join(list_letters_noPunct)

    # Split this big string into a list of words:
    list_words = text_noPunct.strip().split()
    
    # Convert to lower-case letters:
    list_words = [ word.lower() for word in list_words ]
    return list_words



BANNED_LOC = "Data/banned_reddits.txt"
banned_dict = {}
with open(BANNED_LOC) as f:
    for l in f:
        banned_dict[l.strip()[2:]] = True


f = open(CSV_FILE, 'w', newline='')
writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
writer.writerow(["id10","datetime", "subreddit", "author", "body", "banned", "suspect"])

def insert(post):
    writer.writerow([post["id10"], post["created_utc"], post["subreddit"], post["author"], post["body"], post["banned"], post["suspect"]])


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
#    cleaned_text = clean_text(post["body"])
#    post["bodySan"] = ' '.join(cleaned_text)
    insert(post)
    if num_read%int(1e4) == 0:
        print("Read in: %10dK posts in %.2fs (%.2fs)"%(num_read//int(1e3), time.time() - t0, time.time() - t1), flush=True)
        t1 = time.time()

f.close()

