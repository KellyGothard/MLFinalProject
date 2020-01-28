# (setq python-shell-interpreter "~/python-environments/data-science/bin/python")


import json
import csv

threads = []
with open("threads_data_100k.json") as f:
    for line in f:
        js = json.loads(line)
        thread = {"subreddit": js["subreddit"],
                  "banned": js["banned"],
                  "comments": " ".join([comm["body"] for comm in js["comments"]])}
        threads.append(thread)


cols = ["subreddit", "banned", "comments"]
with open("reddit_threads.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(cols)
    for thread in threads:
        writer.writerow([thread[col] for col in cols])
        
