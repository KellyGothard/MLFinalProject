# local editor config
# (setq python-shell-interpreter "~/python-environments/ml/bin/python")


"""
Builds a basic logistic regression classifier to group posts into
either being from a banned subreddit or being from a not banned
subreddit. Puts posts into vector space by using the fastText
(https://fasttext.cc/) word embedding. Fasttext can only make
representations of words, so we get representations of posts by
summing up the word vectors for individual words in the post.

Note: limits the number of words vectors we read into the model to
200,000, as we are limited by ram. Can increase this if you have more
ram, or by using a method to not read the word vectors in from the
model all at once (e.g. database).
"""

MAX_WORDS = 200000

import fasttext
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def load_vectors(fname):
    """read in word vectors from file"""
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = [float(tok) for tok in tokens[1:]]
        # cap on the number of words we can read in. If too high,
        # computer doesn't have enough ram. Can be remedied by either
        # using a computer with more ram, or by making a local
        # database to read from instead of reading in the entire file
        # at once.
        if len(data) > MAX_WORDS:
            break
    fin.close()
    return data, d

# https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
model, d = load_vectors("/home/nick/downloads/fast-wiki/wiki-news-300d-1M.vec")
print(model['king'])


# folder that contains notbanned.csv and banned.csv
folder = "data/"

notbanned_file = folder + "notbanned.csv"
banned_file = folder + "banned.csv"

# read in data
notbanned_df = pd.read_csv(notbanned_file)
banned_df = pd.read_csv(banned_file)

# combine two dataframes, do some cleaning
data = pd.concat([notbanned_df, banned_df], ignore_index=True)[["author", "body", "banned"]]
data["body"] = [(" ".join(str(b).split())).strip() for b in data["body"]]
data = data[data["body"] != "[deleted]"]
data = data.reset_index()

# to hold summed word vectors for each comment
vecs = np.zeros([len(data), d])

# iterate through data, mapping each word that has a mapping to its
# corresponding word vector then normalize that vector and add it to
# the word vector of any other words in the post. do this for each
# word to get some kind of representation of post.
for idx, row in data.iterrows():
    svec = np.zeros(d)
    for word in row["body"].split():
        if word in model:
            wvec = model["word"]
            if len(wvec) == d:
                svec += wvec / np.linalg.norm(wvec)
    vecs[idx] = svec

    
# split into test and train datasets (will implement cross-val and proper testing later)
X_train, X_test, y_train, y_test = train_test_split(vecs, data["banned"], test_size=0.33, random_state=42)

# make and fit logistic regression model
clf = LogisticRegression(solver="lbfgs", n_jobs=-1)
clf.fit(X_train, y_train)

# show results from running model on test set
print(classification_report(clf.predict(X_test), y_test))
