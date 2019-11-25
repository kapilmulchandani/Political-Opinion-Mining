import pandas as pd
import re
import unidecode
import nltk
import string
from nltk.corpus import stopwords
# nltk.download('stopwords')
stop_words = list(set(stopwords.words('english')))

# list of all candidates to be removed from the tweets
candidates = ["bernie", "berni", "sanders", "sander", "joe", "biden", "elizabeth", "warren", "donald", "trump", "mayor", "pete",
                        "buttigieg", "kamala", "harris", "andrew", "yang", "tulsi", "gabbard"]

# words that can be considered for removal in pre-processing
add_stopwords = ['s', 'userhandle', 'user', 'u', 'a', 'amp', 'presidential', '2020', 'democratic', 'president',
                 'democrat', 'us']

stop_words.extend(add_stopwords)
stop_words.extend(candidates)


def lemma(tweet):
    # removal of punctuations
    tweet = re.sub(r'[^\w\s]', '', tweet)
    # print(tweet)

    # removal of stopwords
    words = [word
             for word in tweet.split()
             if word not in stop_words]

    # print(words)
    tweet_stem = ' '.join(words)

    # print(tweet_stem)
    return tweet_stem + " "


def preprocess(tweet):
    # convert to lowercase
    tweet = tweet.lower()

    # Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', tweet)

    # Convert @username to __USERHANDLE
    tweet = re.sub('@[^\s]+', '', tweet)

    # Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

    # trim
    tweet = tweet.strip('\'"')

    # Repeating words like hellloooo
    repeat_char = re.compile(r"(.)\1{1, }", re.IGNORECASE)
    tweet = repeat_char.sub(r"\1\1", tweet)

    # remove non ascii
    tweet = unidecode.unidecode(tweet)

    # removal of punctuations
    tweet = tweet.translate(str.maketrans(' ', ' ', string.punctuation))
    return tweet


# candidate name for word cloud file
candidate_names = ['bernie', 'PeteButtigieg', 'ElizabethWarren', 'JoeBiden', 'KamalaHarris']

# Generating word cloud file for each candidate
for name in candidate_names:
    # candidates' csv file to read
    filename = 'E:/255-Project/Political-Opinion-Mining/Candidate Tweets/selected_' + name + '_tweets.csv'

    # candidates' csv file to write
    words_file = name + '_words.csv'

    data = pd.read_csv(filename, header=None, index_col=0)
    d = [lemma(preprocess(tweet)) for tweet in data[5]] # data[column] where column => column containing tweets

    d = ''.join(d)

    # removal of punctuations and converting to single string
    d = re.sub(r'[^\w\s]', '', d)

    # Convert string to series
    series = pd.Series(d.split())
    # print(series)
    df = pd.DataFrame(series.value_counts()).reset_index()
    print(df)
    df.to_csv(words_file)


