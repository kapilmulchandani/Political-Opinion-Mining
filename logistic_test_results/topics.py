import pandas as pd

candidates = {'bernie': 'Bernie Sanders', 'pete': 'Pete Buttigieg', 'harris': 'Kamala Harris',
              'warren': 'Elizabeth Warren', 'biden': 'Joe Biden'}

topics = {'economy': 'Economy', 'healthcare': 'Health Care', 'gun control': 'Gun Control', 'immigration': 'Immigration',
          'taxes': 'Taxes'}
rows = []

for name in candidates.keys():
    for topic in topics.keys():
        data = pd.read_csv('tweets_' + name + '_' + topic + '.csv', header=None, index_col=0)

        data = data.iloc[:, [0, 3]]
        data.columns = ['Sentiment', 'Date']
        print(data.head())
        dictionary = dict(data['Sentiment'].value_counts())
        total_len = len(data)
        pos = round((dictionary[4]/total_len) * 100, 2)
        neg = round((dictionary[0]/total_len) * 100, 2)

        # For topic wise candidate sentiment
        row = {'Name': candidates[name], 'Topic': topics[topic], 'POS': dictionary[4], 'NEG': dictionary[0],
               'Total': total_len, 'POS%': pos, 'NEG%': neg}
        # print(row)
        rows.append(row)

df = pd.DataFrame(rows)
print(df)
# df.to_csv('topicwise_sentiment.csv')
# df.to_csv('overall_sentiment for candidates.csv')
# df.to_csv('locations_sentiment.csv')

# df_max.to_csv('most_tweeted_candidate_location.csv')