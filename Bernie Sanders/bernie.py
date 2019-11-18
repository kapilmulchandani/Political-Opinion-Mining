# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 23:32:55 2019

@author: prach
"""

import tweepy
import csv
import os


# Fill this
access_token = ""
access_token_secret = ""
consumer_key = ""
consumer_secret = ""
CANDIDATE = 'bernie'


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)

queries = {
    'bernie': ['bernie sanders'],
    'warren': ['elizabeth warren'],
    'biden': ['joe biden'],
    'pete': ['pete buttigieg', 'mayor pete'],
    'harris': ['kamala harris'],        
}

geocodes = {
    'sf': (37.783333, -122.416667),
    'nyc': (40.730610, -73.935242),
    'houston': (29.749907, -95.358421),
    'detroit': (42.331429, -83.045753),
    'seattle': (47.608013, -122.335167),
    'miami':(25.761681,	-80.191788)
}

def get_tweets(kwds, output_file, limit=100000, lang='en', until=None,
                count=100, coords=None):
    query = ' OR '.join(kwds) + ' -filter:retweets'
    geo = f'{coords[0]},{coords[1]},100km' if coords else None
    print('Searching for', query, 'in', geo)
    
    max_id = 0
    if os.path.exists(output_file):
        with open(output_file,encoding='utf-8-sig') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                if len(row) > 0:
                    max_id = max(max_id, int(row[1]))
        print('Max tweet id =', max_id)

    cursor = tweepy.Cursor(
        api.search,
        q=query,
        count=count,
        lang=lang,
        geocode=geo,
        since_id=max_id
    )
    with open(output_file, 'a',encoding='utf-8-sig',newline='') as csvFile:
        csvWriter = csv.writer(csvFile)
        for count, tweet in enumerate(cursor.items()):
#            print(tweet.id)
            text = tweet.text.replace('\n', ' ')
            csvWriter.writerow([
                query,
                tweet.id,
                tweet.created_at,
                tweet.user.screen_name,
                tweet.geo,
                tweet.author.location,
                text,
            ])
            if count == limit - 1:
                break


if __name__ == '__main__':
    name = CANDIDATE
    kwds = queries[CANDIDATE]
    get_tweets(kwds, f'tweets_{name}.csv', limit=10000)
    
#    # Geo-coded tweets
    for loc, coords in geocodes.items():
        get_tweets(kwds, f'tweets_{name}_{loc}.csv', limit=10000,
                   coords=coords)

