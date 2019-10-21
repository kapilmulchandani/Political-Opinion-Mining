import pandas as pd

data = pd.read_json("json/tweets.json", lines=True)
data.to_csv("csv/tweets.csv")
data = pd.read_csv("csv/tweets.csv")

data['user'] = data['user'].apply(eval)
print(type(data['user'][0]))

json_list = data['user'].to_json("json/user.json", orient="records")
json_df = pd.read_json("json/user.json")
print(json_df.head())

json_df.to_csv("csv/user.csv")