
from datetime import timedelta
import ujson
import pandas as pd
from datetime import datetime
import praw
import re
import requests
import pprint
import time
import math

def process(df,vectorizer,stemmer):
    punctuations = '''!()\-[]{};:'"\,<>./?@#$%^&*_~|''';
    documents = []
    posts = list(df['body'])
    s = ''
    for post in posts:
        for char in punctuations:
            post = post.replace(char, '')
        posttext = post.replace('\n','') + ' '
        s += posttext
        document = re.sub(r'^https?:\/\/.*[\r\n]*', '', s, flags=re.MULTILINE)
        document = document.lower()
        document = document.split()
        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)
        documents.append(document)
    v = vectorizer.fit_transform(documents).toarray()
    return v

def make_request(uri, max_retries = 5):
    def fire_away(uri):
        response = requests.get(uri)
        assert response.status_code == 200
        return ujson.loads(response.content)
    current_tries = 1
    while current_tries < max_retries:
        try:
            time.sleep(1)
            response = fire_away(uri)
            return response
        except:
            time.sleep(1)
            current_tries += 1
    return fire_away(uri)

def pull_posts_for(subreddit, start_at, end_at):
    def map_posts(posts):
        return list(map(lambda post: {
            'id': post['id'],
            'created_utc': post['created_utc'],
            'body': post['body'],
            'author':post['author']
        }, posts))
    
    SIZE = 500
    URI_TEMPLATE = r'https://api.pushshift.io/reddit/search/comment?subreddit={}&after={}&before={}&size={}'
    
    post_collections = map_posts( \
        make_request( \
            URI_TEMPLATE.format( \
                subreddit, start_at, end_at, SIZE))['data'])
    n = len(post_collections)
    while n == SIZE:
        last = post_collections[-1]
        new_start_at = last['created_utc'] - (10)
        
        more_posts = map_posts( \
            make_request( \
                URI_TEMPLATE.format( \
                    subreddit, new_start_at, end_at, SIZE))['data'])
        
        n = len(more_posts)
        post_collections.extend(more_posts)
    return post_collections

def give_me_intervals(start_at, number_of_days_per_interval = 3):
    
    end_at = math.ceil(datetime.utcnow().timestamp())
        
    period = (86400 * number_of_days_per_interval)
    end = start_at + period
    yield (int(start_at), int(end))
    padding = 1
    while end <= end_at:
        start_at = end + padding
        end = (start_at - padding) + period
        yield int(start_at), int(end)

def main():
    
    # PRAW
    reddit = praw.Reddit(client_id='gz2ObTGldvgEMg',
                         client_secret='6r1S2-WPPhyQDJmNuh8aW-b1eWY',
                         user_agent='project')
    
    #SR_random = reddit.subreddit('random')
    SR_test = 'duvetdudes'

    start_at = math.floor(\
        (datetime.utcnow() - timedelta(days=365)).timestamp())
    posts = []
    for interval in give_me_intervals(start_at, 7):
        pulled_posts = pull_posts_for(
            SR_test, interval[0], interval[1])
        
        posts.extend(pulled_posts)
        time.sleep(.500)

    df = pd.DataFrame(posts)
    print(df.columns)

if __name__=="__main__":
    main()
