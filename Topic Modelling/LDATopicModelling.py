from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import nltk
from nltk.corpus import stopwords
import pandas as pd

nltk.download('stopwords')
stopwords = stopwords.words('english')
candidates = ['bernie', 'sanders', 'elizabeth', 'warren', 'pete', 'mayor', 'joe', 'biden', 'kamala', 'harris', 'buttigieg']
stopwords.extend(candidates)
tokenizer = RegexpTokenizer(r'\w+')

allTweetsdf = pd.read_csv("/home/kapil/PycharmProjects/Political-Opinion-Mining/Candidate_AND_Every_Other_Topic_Query_Tweets/all_candidates_tweets.csv")
# create English stop words list
# en_stop = stopwords

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

doc_set = []
for i in range(0, allTweetsdf.shape[0]):
    # print(allTweetsdf.iloc[i]['text'])

    doc_set.append(allTweetsdf.iloc[i]['text'])

# compile sample documents into a list
# doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]

# list for tokenized documents in loop
texts = []

# loop through document list
for i in doc_set:

    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in stopwords]

    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

    # add tokens to list
    texts.append(stemmed_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=4, id2word = dictionary, passes=20)

print(ldamodel.print_topics(num_topics=2, num_words=4))

print(ldamodel.print_topics(num_topics=3, num_words=3))