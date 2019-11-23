import sqlite3
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import numpy as np
from collections import deque
import pickle
import csv

print("import done",  flush=True)
fname = "/users/d/m/dmatthe1/OtherProjects/DBs/RC_2016-10.db"
pickle_name = "/users/d/m/dmatthe1/OtherProjects/DBs/2016-10-Data_Rand_Comments.p"
CSV_NAME = "/users/d/m/dmatthe1/OtherProjects/DBs/2016-10-Data_Rand_Comments.csv"

DATA_POINTS = int(4e5)

DOC_SIZE = 200
tokenizer = RegexpTokenizer(r'\w+')
print("beginning to load posts",  flush=True)
try:
    print("Selecting posts")
    tmp_banned, tmp_not_banned = pickle.load(open(pickle_name, "rb"))
    print("successfully read from pickle")
except:
    print("Using sqlite", flush=True)
    conn = sqlite3.connect(fname)
    c = conn.cursor()
    c.execute('SELECT  body FROM Comments WHERE banned == "True"  and suspect == "False" ORDER BY RANDOM() LIMIT %d'%DATA_POINTS) # ORDER BY RANDOM()
    print("1/2 queried")
    tmp_banned = c.fetchall()
    print("1/2 read")

    c.execute('SELECT  body FROM Comments WHERE banned == "False" and suspect == "False" ORDER BY RANDOM() LIMIT %d'%DATA_POINTS) # ORDER BY RANDOM()
    print("2/2 queried")
    tmp_not_banned = c.fetchall()
    print("2/2 read")

    pickle.dump((tmp_banned, tmp_not_banned), open(pickle_name, "wb"))
    print("cached.")

print("Posts loaded")
banned_res = deque()
print("Tokenizing... 1/2")
for r in tmp_banned:
    for word in tokenizer.tokenize(r[0]):
        banned_res.append(word)

banned_res = list(banned_res)
TRIM = -1 * (len(banned_res)%DOC_SIZE)
banned_res  = banned_res[:TRIM]

banned_arr = [None]*(len(banned_res)//DOC_SIZE)
for i in range(len(banned_res)//DOC_SIZE):
    banned_arr[i] = banned_res[i*DOC_SIZE : (i+1)*DOC_SIZE]

not_banned_res = deque()
print("Tokenizing... 2/2")
for r in tmp_not_banned:
    for word in tokenizer.tokenize(r[0]):
        not_banned_res.append(word)

print("Posts tokenized")
not_banned_res = list(not_banned_res)
TRIM = -1 * (len(not_banned_res)%DOC_SIZE)
not_banned_res  = not_banned_res[:TRIM]

not_banned_arr = [None]*(len(not_banned_res)//DOC_SIZE)
for i in range(len(not_banned_res)//DOC_SIZE):
    not_banned_arr[i] = not_banned_res[i*DOC_SIZE : (i+1)*DOC_SIZE]

print("saving to CSV")
writer = csv.writer(open(CSV_NAME, "w"), quoting=csv.QUOTE_MINIMAL)
writer.writerow(["document", "banned"])
for row in banned_arr:
    writer.writerow([' '.join(row), 1])
for row in not_banned_arr:
    writer.writerow([' '.join(row), 0])

print("banned comments")
for row in banned_arr[:10]:
    print(row)

print("not banned comments")
for row in not_banned_arr[:10]:
    print(row)


print("Found %d banned comments."%len(tmp_banned))
print("Merged to form %d documents."%len(banned_arr))

print("Found %d not banned comments."%len(tmp_not_banned))
print("Merged to form %d documents."%len(not_banned_arr))
