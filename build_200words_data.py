import bz2
import json
import random
from time import time
import csv
from nltk.tokenize import word_tokenize


def get_subreddit_data(filename, banned_subreddits, max_comments=100000):
    comments = dict()
    
    with bz2.BZ2File(filename, "r") as f:
        i = 0
        for line in f:
            if i >= max_comments:
                break
            i += 1
            
            comm = json.loads(line.decode())
            subreddit = comm["subreddit"]
            trimmed_comm = {"subreddit": subreddit,
                            "banned": subreddit in banned_subreddits,
                            "body": comm["body"]}
            if subreddit in comments:
                comments[subreddit].append(trimmed_comm)
            else:
                comments[subreddit] = [trimmed_comm]
    return comments



def read_banned_subreddits(filename):
    """read in banned subreddits file with each line as 'r/asfasdf' """
    with open(filename, "r") as f:
        banned = set()
        for line in f:
            if line.strip() != "":
                banned.update([line.strip().split("r/")[1]])
        return banned

def flatten(l):
    return [item for sublist in l for item in sublist]

def divide_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def chunkify_train(train, words_per_example):
    banned_comments = [comm for comm in train if comm["banned"]]
    notbanned_comments = [comm for comm in train if not comm["banned"]]

    banned_stream = flatten([word_tokenize(comm["body"]) for comm in banned_comments])
    notbanned_stream = flatten([word_tokenize(comm["body"]) for comm in notbanned_comments])

    banned_examples = list(divide_chunks(banned_stream, words_per_example))
    notbanned_examples = list(divide_chunks(notbanned_stream, words_per_example))

    return banned_examples, notbanned_examples

def chunkify_subreddit(subreddit, words_per_example):
    banned = subreddit[0]["banned"]
    my_subreddit = subreddit[0]["subreddit"]
    
    stream = flatten([word_tokenize(comm["body"]) for comm in subreddit])
    chunks = list(divide_chunks(stream, words_per_example))
    return [{"subreddit": my_subreddit, "banned": banned, "words": chunk} for chunk in chunks]

def chunkify_test(test, words_per_example=200):
    subreddit_to_comments = dict()
    for comm in test:
        if comm["subreddit"] in subreddit_to_comments:
            subreddit_to_comments[comm["subreddit"]].append(comm)
        else:
            subreddit_to_comments[comm["subreddit"]] = [comm]

    examples = flatten([chunkify_subreddit(subreddit, words_per_example)
                        for _, subreddit in subreddit_to_comments.items()])
    return examples

    

def split_data(subreddit_to_comments, ratio, words_per_example=200):
    random.seed(42)



    flattened = flatten([comments for _, comments in subreddit_to_comments.items()])
    random.shuffle(flattened)

    split_idx = int(len(flattened) * ratio)
    
    train = flattened[:split_idx]
    test = flattened[split_idx:]

    train_banned, train_notbanned = chunkify_train(train, words_per_example)
    test_examples = chunkify_test(test, words_per_example)
    return train_banned, train_notbanned, test_examples
    

def write_train_examples(examples, filename):
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["words"])
        for example in examples:
            writer.writerow([" ".join(example)])


def write_test_examples(examples, filename):
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["subreddit", "banned", "words"])
        for example in examples:
            writer.writerow([example["subreddit"], example["banned"], " ".join(example["words"])])

    
data_folder = "/home/nick/downloads/reddit/"

banned_subreddits = read_banned_subreddits(data_folder + "banned-subreddits.txt")
subreddit_to_comments = get_subreddit_data(data_folder + "RC_2016-10.bz2", banned_subreddits, max_comments=100000)
train_banned, train_notbanned, test_examples = split_data(subreddit_to_comments, 0.8)

write_train_examples(train_banned, "banned_200.csv")
write_train_examples(train_notbanned, "notbanned_200.csv")
write_test_examples(test_examples, "test_200.csv")
