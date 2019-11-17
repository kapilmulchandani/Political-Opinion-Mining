import tweepy
import csv
import twitter_credentials as tc

auth = tweepy.OAuthHandler(tc.CONSUMER_KEY, tc.CONSUMER_SECRET)
auth.set_access_token(tc.ACCESS_TOKEN, tc.ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)  # Open/Create a file to append data
csvFile = open('tweets4.csv', 'a') #Use csv Writer
csvWriter = csv.writer(csvFile)
search_term = "bernie sanders"
for tweet in tweepy.Cursor(api.search, q=search_term, count=100, lang="en", since_id=2019 - 11 - 16).items():
    print(tweet.created_at, tweet.text)
    csvWriter.writerow([tweet.id, tweet.created_at, tweet.user.screen_name.encode('utf8'), tweet.text.encode('utf8')])
