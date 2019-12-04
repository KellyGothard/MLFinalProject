from sklearn.model_selection import train_test_split
import pandas as pd

# How many words per training example?
SEQLEN = 200

# How many training examples of each type?
bannedCnt = 50000
notbannedCnt = 50000

# Feather save locations
SAVE_FILES = "../Data/Generated/RC_2016-10_%s.pkl"

# where are the source files?
# These are generated by PreProcessComments.py
banned_txt = "../Data/Generated/RC_2016-10_banned.txt"
notbanned_txt = "../Data/Generated/RC_2016-10_notbanned.txt"

# open the source files.
bannedF = open(banned_txt,"r")
notbannedF = open(notbanned_txt, "r")

# log progress.
print("Will load %d banned examples and %d notbanned examples"%(bannedCnt, notbannedCnt))

# load the training examples
all_posts = []
bannedPosts = [(1, [bannedF.readline().strip().lower() for _ in range(SEQLEN)]) for _ in range(bannedCnt)]
notbannedPosts  = [(0, [notbannedF.readline().strip().lower() for _ in range(SEQLEN)]) for _ in range(notbannedCnt)]

all_posts += bannedPosts
all_posts += notbannedPosts

print("Train Test Split...")
# split to have about 10K in the test set.
train, test = train_test_split(all_posts, test_size=0.1)


print("Generating Data Frames")
dfTrain= pd.DataFrame(train, columns=["banned", "tokens"])
dfTest = pd.DataFrame(test, columns=["banned", "tokens"])

print(dfTrain.head())
print(dfTest.head())

# save dataframes
print("Saving Data Frames to file")
dfTrain.to_pickle(SAVE_FILES%"Train")
dfTest.to_pickle(SAVE_FILES%"Test")