import pandas as pd

candidates = {'bernie': 'Bernie Sanders', 'pete': 'Pete Buttigieg', 'harris': 'Kamala Harris',
              'warren': 'Elizabeth Warren', 'biden': 'Joe Biden'}

topics = {'economy': 'Economy', 'healthcare': 'Health Care', 'gun control': 'Gun Control', 'immigration': 'Immigration',
          'taxes': 'Taxes'}

df = pd.DataFrame()

for name in candidates:
    for topic in topics.keys():
        filename = 'tweets_' + name + '_' + topic + '.csv'
        data = pd.read_csv(filename, index_col=0, header=None)
        data = data[4].str.slice(0, 10)
        data = data.value_counts().reset_index()
        data.columns = ['Date', 'Count']
        data['Topic'] = topics[topic]
        data['Candidate'] = candidates[name]
        # print(data)
        df = df.append(data)

# df.to_csv('topics_datewise.csv')
# print(len(df))