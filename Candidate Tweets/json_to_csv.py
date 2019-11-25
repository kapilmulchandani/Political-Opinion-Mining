import pandas as pd

candidate_names = ['bernie', 'PeteButtigieg', 'ElizabethWarren', 'JoeBiden', 'KamalaHarris']
dataframe_list = []

for name in candidate_names:
    data = pd.read_json("E:/255-Project/Political-Opinion-Mining/Candidate Tweets/" + name + "_tweets.json")
    data.to_csv(name + '_tweets.csv')
    data = pd.read_csv(name + '_tweets.csv', index_col=0)
    columns = ['username', 'likes', 'replies', 'retweets', 'text', 'timestamp']
    data = data[columns]
    # print(data.head())
    data.to_csv("selected_" + name + "_tweets.csv")
    dataframe_list.append(data)

merged_df = pd.concat(dataframe_list).reset_index()
merged_df = merged_df.drop(columns=['index'])
print(merged_df)
merged_df.to_csv('all_candidates_tweets.csv')