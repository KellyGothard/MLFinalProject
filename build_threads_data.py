from compile_threads_utils import rebuild_threads, read_banned_subreddits, threads_to_file

import numpy as np
import os

folder = "/home/nick/downloads/reddit"

# rebuild threads
grandparent2children, comments = rebuild_threads(os.path.join(folder, "RC_2016-10.bz2"),
                                                 max_comments=100000)

# get banned subreddits
banned_subreddits = read_banned_subreddits(os.path.join(folder, "banned-subreddits.txt"))

# write data to file
threads_to_file(grandparent2children, comments, banned_subreddits, "threads_data_100k.json")
