import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

data = pd.read_json("json/tweets.json", lines=True)
data.to_csv("csv/tweets.csv")
data = pd.read_csv("csv/tweets.csv")

data['user'] = data['user'].apply(eval)
print(type(data['user'][0]))

json_list = data['user'].to_json("json/user.json", orient="records")
json_df = pd.read_json("json/user.json")

stop_words = set(stopwords.words('english'))
for i in range (0, json_df.shape[0]):
    print(json_df['description'][i])
    word_tokens = word_tokenize(json_df['description'][i])
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    str1 = " ".join(filtered_sentence)
    print(str1)
    json_df['description'][i] = str1

print(json_df['description'])

json_df.to_csv("csv/user.csv")