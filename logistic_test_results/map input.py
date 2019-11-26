import pandas as pd

candidates = {'bernie': 'Bernie Sanders', 'pete': 'Pete Buttigieg', 'harris': 'Kamala Harris',
              'warren': 'Elizabeth Warren', 'biden': 'Joe Biden'}

locations = {'detroit': 'Detroit', 'sf': 'San Francisco', 'miami': 'Miami', 'nyc': 'New York', 'houston': 'Houston',
             'seattle': 'Seattle'}

rows = []

for name in candidates.keys():
    for location in locations.keys():
        data = pd.read_csv('tweets_' + name + '_' + location + '.csv', header=None, index_col=0)

        data = data.iloc[:, [0, 3]]
        data.columns = ['Sentiment', 'Date']
        # print(data.head())
        # data[4] = pd.to_datetime(data[4], format="%m/%d/%Y")
        # exit(0)
        dictionary = dict(data['Sentiment'].value_counts())

        total_len = len(data)
        pos = round((dictionary[4]/total_len) * 100, 2)
        neg = round((dictionary[0]/total_len) * 100, 2)

        # For overall all candidate tweets
        # row = {'Name': candidates[name], 'POS': dictionary[4], 'NEG': dictionary[0], 'Total': total_len, 'POS%': pos,
        # 'NEG%': neg}

        # For location wise candidate sentiment
        row = {'Name': candidates[name], 'Location': locations[location], 'POS': dictionary[4], 'NEG': dictionary[0],
               'Total': total_len, 'POS%': pos, 'NEG%': neg}
        print(row)
        rows.append(row)

df = pd.DataFrame(rows)
print(df)
# df.to_csv('overall_sentiment for candidates.csv')
# df.to_csv('locations_sentiment.csv')
print(df.groupby(by=['Location']).max())
id = df.groupby(by=['Location'])['Total'].idxmax()
df_max = df.loc[id,]
print(df_max)

# df_max.to_csv('most_tweeted_candidate_location.csv')