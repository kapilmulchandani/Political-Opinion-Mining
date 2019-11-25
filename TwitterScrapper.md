Command for getting tweets based on user:
twitterscraper "from:PeteButtigieg" -o PeteButtigieg_tweets.json -l 1000 -bd 2019-10-01 -ed 2019-11-23


twitterscraper Trump -l 10 -bd 2019-10-20 -ed 2017-11-23 -o tweets1monthlatest.json

twitterscraper "Sanders AND economy OR immigration OR Health Care OR Medicare OR Gun Control OR Taxes" -l 10 -bd 2019-10-23 -ed 2019-11-16 -o SandersAndEveryTopic.json
twitterscraper "Elizabeth Warren AND economy OR immigration OR Health Care OR Medicare OR Gun Control OR Taxes" -l 10 -bd 2019-10-23 -ed 2019-11-16 -o WarrenAndEveryTopic.json

twitterscraper "Warren AND economy OR immigration OR Health Care OR Medicare OR Gun Control OR Taxes" -l 50000000 -bd 2019-10-23 -ed 2019-11-16 -o WarrenAndEveryTopic.json

twitterscraper "Warren AND economy OR immigration OR Health Care OR Medicare OR Gun Control OR Taxes" -l 50000000 -bd 2019-09-23 -ed 2019-11-23 -o WarrenAndEveryTopic.json
twitterscraper "Biden AND economy OR immigration OR Health Care OR Medicare OR Gun Control OR Taxes" -l 50000000 -bd 2019-09-23 -ed 2019-11-23 -o BidenAndEveryTopic.json
twitterscraper "Pete AND economy OR immigration OR Health Care OR Medicare OR Gun Control OR Taxes" -l 50000000 -bd 2019-09-23 -ed 2019-11-23 -o PeteAndEveryTopic.json
twitterscraper "Kamala AND economy OR immigration OR Health Care OR Medicare OR Gun Control OR Taxes" -l 50000000 -bd 2019-09-23 -ed 2019-11-23 -o KamalaAndEveryTopic.json

twitterscraper "Blockchain near:Seattle within:15mi" -o blockchain_tweets.json -l 1000

twitterscraper "Bernie Sanders near:California" -bd 2019-09-23 -ed 2019-11-23 -o Bernie_California.json



Todos:

LDA to get most popular tweets
<!-- Get twitter data for each state and for each Candidate -->
Labelling
SVM
