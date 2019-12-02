# (setq python-shell-interpreter "~/python-environments/ml/bin/python")


import pandas as pd
import os
import bz2
import json
import random
from time import time
import csv


import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression 








def rebuild_threads(filename, max_comments=2000, update_num=1000, debug=False):
    parent2children = dict()
    comments = dict()

    
    with bz2.BZ2File(filename, "r") as f:
        i = 0
        t0 = time()
        for line in f:
        # for comm in test:
            if i % update_num == 0:
                print(f"Comments complete: {i}/{max_comments}. "
                      f"Time for last {update_num}: {time() - t0:.2f} s")
                t0 = time()
            if i < max_comments:
                comm = json.loads(line.decode())

                my_parent = comm["parent_id"]
                my_id = "t1_" + comm["id"]


                if my_parent in comments:

                    if debug:
                        print(my_id, "has parent", my_parent, "which exists in graph")

                    found = False

                    my_grandparent = None

                    # see if we are a direct child of any parents
                    for (parent, grandparent), children in parent2children.items():
                        if parent == my_parent:

                            if debug:
                                print(my_id, "found parent", my_parent, ", adding", my_id, "to children")
                            
                            parent2children[(parent, grandparent)].update([my_id])
                            my_grandparent = grandparent
                            found = True
                            break

                    # not a direct child of anyone, so need to look through 
                    if not found:
                        for (parent, grandparent), children in parent2children.items():
                            if my_parent in children:

                                if debug:
                                    print(my_id, "found parent", my_parent, "as a child of", grandparent)

                                parent2children[(parent, grandparent)].update([my_id])
                                my_grandparent = grandparent
                                my_parent = parent

                                break

                    # scan for whether we are the grandparent of anyone
                    p2c_new = parent2children.copy()

                    if debug:
                        print("looking for children of", my_id, "already in graph")

                    for (parent, grandparent), children in parent2children.items():
                        if grandparent == my_id:
                            if debug:
                                print("found children", children, "with parent", parent, "of",
                                      my_id, "whose parent is", grandparent)

                            p2c_new[(my_parent, my_grandparent)].update([*children, parent])
                            p2c_new.pop((parent, grandparent))
                    parent2children = p2c_new

                    if debug:
                        print("done looking for children of", my_id)

                # our parent is not in the structure yet
                else:
                    if debug:
                        print(my_id, "has parent", my_parent, "which is not in graph. Making new node")
                        print("looking for children of", my_id, "already in graph")

                    my_children = set()
                    
                    p2c_new = parent2children.copy()
                    for (parent, grandparent), children in parent2children.items():
                        if grandparent == my_id:
                            if debug:
                                print("found children", children, "of", my_id,
                                      "whose parent is", grandparent)
                            my_children.update([*children, parent])
                            p2c_new.pop((parent, grandparent))

                    parent2children = p2c_new
                    parent2children[(my_id, my_parent)] = my_children

                    if debug:
                        print("done looking for children of", my_id)

                # add the comment to our list of comments
                comments[my_id] = comm

            else:
                break
            i += 1


        grandparent2children = top_level_reduce(parent2children)

    return grandparent2children, comments

def top_level_reduce(parent2children):
    """given a dictionary mapping (parent, grandparent) -> set(children),
    return a mapping of grandparent -> set(children). Essentially
    condenses the top level of the graph to group together comments with a
    common ancestor that we dont have access to"""
    grandparent2children = dict()
    for (parent, grandparent), children in parent2children.items():
        if grandparent in grandparent2children:
            grandparent2children[grandparent].update([*children, parent])
        else:
            grandparent2children[grandparent] = set([*children, parent])
    return grandparent2children




# test cases
def build_test_file(filename):
    """build a test file to test the functionality of rebuild_threads.
    Returns what grandparent2children should be when returned from
    rebuild_threads"""
    test = [("a", "z"), ("b", "a"), ("e", "b"), ("f", "b"), ("g", "b"), ("c", "y"), ("d", "y"), ("h", "d"), ("m", "n"), ("j", "e"), ("k", "j")]
    random.shuffle(test)
    test = [{"id": x, "parent_id": "t1_" + y} for (x,y) in test]
    with bz2.BZ2File(filename, "w") as f:
        for comm in test:
            f.write(json.dumps(comm).encode())
            f.write("\n".encode())
    goal = {'t1_y': {'t1_d', 't1_h', 't1_c'},
            't1_z': {'t1_e', 't1_k', 't1_b', 't1_a', 't1_g', 't1_j', 't1_f'},
            't1_n': {'t1_m'}}
    return goal

def test_rebuild_threads():
    """run the test of rebuild_threads"""
    filename = "test.json.bz2"
    goal = build_test_file(filename)
    grandparent2children, comments = rebuild_threads(filename)
    if grandparent2children == goal:
        print("Test passed")
    else:
        print("Test failed")


def confirm_threads_are_contained(grandparent2children):
    """check that for each thread, all comments are in same subreddit.
    Another verification that rebuild_threads is working as intended"""
    for grandparent, children in list(grandparent2children.items())[:10]:
        my_comments = [comments[child] for child in children]
        all_same = True
        subreddit = my_comments[0]["subreddit"]
        print("new:", subreddit)
        for comm in my_comments:
            print(comm["subreddit"])
            if comm["subreddit"] != subreddit:
                all_same = False
                break
        if not all_same:
            print("uh oh")


def read_banned_subreddits(filename):
    """read in banned subreddits file with each line as 'r/asfasdf' """
    with open(filename, "r") as f:
        banned = set()
        for line in f:
            if line.strip() != "":
                banned.update([line.strip().split("r/")[1]])
        return banned

    
def threads_to_file(grandparent2children, comments, banned_subreddits, filename):
    """Write threads to file. Writes a json lines file with each object
    having the keys 'comments', 'subreddit', 'banned', and 'parent'.
    Comments is a string of json of all comments belonging to the
    parent, subreddit is a string of the name of the subreddit, banned
    is a bool representing whether the thread belongs to a banned
    subreddit or not, and parent is the id of the top level
    comment/link to which all the comments belong. 

    Prints out the number of banned and not banned threads.
    """
    n_banned = 0
    n_not_banned = 0
    with open(filename, "w") as f:
        for grandparent, children in grandparent2children.items():
            my_comments = [comments[child] for child in children]
            subreddit = my_comments[0]["subreddit"]
            banned = subreddit in banned_subreddits
            if banned:
                n_banned += 1
            else:
                n_not_banned += 1
            f.write(json.dumps({"comments": my_comments, "subreddit": subreddit, "banned": banned, "parent": grandparent}))
            f.write("\n")
    print("n not banned:", n_not_banned)
    print("n banned:", n_banned)

            
        
            
def time_building_threads(max_comment_nums):
    times = []
    for max_num in max_comment_nums:
        print(f"Testing time for {max_num} max comments")
        t0 = time()
        grandparent2children, comments = rebuild_threads("/home/nick/downloads/RC_2016-10.bz2", max_num)
        diff = time() - t0
        times.append(diff)
    return times



def main():
    banned_subreddits = read_banned_subreddits("/home/nick/downloads/reddit/banned-subreddits.txt")
    threads_to_file(grandparent2children, comments, banned_subreddits, "/home/nick/downloads/reddit/threads.json")

    # time building threads
    max_comment_nums = [2000, 3000, 4000, 5000, 8000, 16000, 32000, 64000]
    times = time_building_threads(max_comment_nums)

    # build regression model to estimate how long it would take to build more comments 
    poly = PolynomialFeatures(degree = 3)
    X_poly = poly.fit_transform([[max_num] for max_num in max_comment_nums])
    lin = LinearRegression()
    lin.fit(X_poly, times)

    # plot tested sizes and estimated model 
    plt.scatter(max_comment_nums, np.array(times) / 60, s=50)
    xs = list(np.linspace(0,500000, 30))
    plt.plot(xs, lin.predict(poly.transform([[x] for x in xs])) / 60)
    plt.xlabel("Number of comments")
    plt.ylabel("Minutes to build threads")
    plt.show()

    # t1_sum = 0
    # t3_sum = 0
    # for grandparent, children in grandparents2children.items():
    #     parent = grandparent.split("_")
    #     if parent[0] == "t1":
    #         t1_sum += 1
    #     elif parent[0] == "t3":
    #         t3_sum += 1
    #     else:
    #         print(parent[0])
    # print("t1:", t1_sum)
    # print("t3:", t3_sum)

    # ccdf on loglog scale for number of comments in thread
    n_children = sorted([len(children) for _, children in grandparent2children.items()])
    probs = 1 - np.arange(len(n_children)) / len(n_children)
    plt.loglog(n_children, probs)
    plt.show()



    # with bz2.BZ2File("/home/nick/downloads/RC_2016-10.bz2", "r") as f:
        # i = 0
        # for line in f:
            # i += 1

    # print(i)

    # print(grandparent2children == goal)


    
