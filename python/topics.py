import csv
import operator
import pickle
import re

import pandas as pd
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords

from python.Features import loadDataFromCsv


def preprocess_text(text):
    text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
    text = re.sub(' +', ' ', text)
    return text.strip()

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

model_ak = pickle.load(open('/Users/hochen/Documents/Projects/prd-review/717283066_model.pkl','rb'))

df = loadDataFromCsv('/Users/hochen/Documents/Projects/prd-review/717283066_sort.csv', rows=3000)
df = df[df['parent_tag_name'] == 'Fashion']

topic_keys = {'delivery':['early', 'arrive'],
        'size':['size', 'fit', 'big'],
        'fabric':['thin',  'cotton'],
        'style': ['cute', 'sporty'],
        'color': ['dark'],
        'occasion': ['concert', 'wedding']
 }

sentiment_keys = {
    'positive_sentiment': ['excellent'],
    'negative_sentiment': ['horrible']}

def topicAnalysis(row):
    scores = {}
    word_count = 0
    for sentence in nltk.sent_tokenize(row['comment']):
        sentence = preprocess_text(sentence.lower())
        words = sentence.split(' ')
        words = list(filter(lambda w:  w not in stop_words, words))
        word_count += len(words)

        topic_score_dict = {}
        for topic, seeds in topic_keys.items():
            topic_score = 0
            for word in words:
                score = max(
                    [(model_ak.wv.similarity(word, seed) if word in model_ak.wv.vocab else 0) for seed in seeds])
                topic_score += (score if score > 0 else 0)

            topic_score_dict[topic] = topic_score

        top_topic = sorted(topic_score_dict.items(), key=operator.itemgetter(1), reverse=True)

        for i in [0, 1]:
            if i == 0 or top_topic[i][1] > 2.0:  # the first topic or if second topic is strong enough
                if top_topic[i][0] not in scores.keys():
                    scores[top_topic[i][0]] = [top_topic[i][1]]
                else:
                    scores[top_topic[i][0]].append(top_topic[i][1])

    topic_list = []
    for topic, val in scores.items():
        if max(val) > 0.7:
            topic_list.append(topic)
        scores[topic] = max(val)

    return pd.Series({'word_count':word_count, 'topic_count': len(topic_list), 'topic_list':topic_list, 'topic_score': f"{scores}"}) #word_count, topic_list, f"{scores}"

def sentimentAnalysis(row):
    scores = {}
    sentiment_score_dict = {}

    for sentence in nltk.sent_tokenize(row['comment']):
        sentence = preprocess_text(sentence.lower())
        words = sentence.split(' ')


        for sentiment, seeds in sentiment_keys.items():
            sentiment_score = 0
            for word in words:
                score = max(
                    [(model_ak.wv.similarity(word, seed) if word in model_ak.wv.vocab else 0) for seed in seeds])
                sentiment_score += (score if score > 0 else 0)

            sentiment_score_dict[sentiment] = sentiment_score

        top_sentiment = sorted(sentiment_score_dict.items(), key=operator.itemgetter(1), reverse=True)[0]

        if top_sentiment[0] not in scores.keys():
            scores[top_sentiment[0]] = [top_sentiment[1]]
        else:
            scores[top_sentiment[0]].append(top_sentiment[1])

    topic_list = []
    for topic, val in scores.items():
        if max(val) > 0.7:
            topic_list.append(topic)
        scores[topic] = max(val)

    return pd.Series({'sentiment':scores})


topic_column = df.apply(topicAnalysis, axis=1)
#sentiment_column = df.apply(sentimentAnalysis, axis=1)

df = pd.concat([df[['product_id','transaction_id','rating_id','rating','comment','image_count']], topic_column], axis=1) #, sentiment_column

features = ['word_count','topic_count']

mean_series = df[features].mean()
print("mean_series: "+str(mean_series))
max_series = df[features].max()
print("max_series: "+str(max_series))
min_series = df[features].min()
print("min_series: "+str(min_series))

range_series = pd.DataFrame([max_series - mean_series, mean_series - min_series]).max()
weight_series = pd.Series({'word_count_norm':0.1, 'topic_count_norm': 0.5}) #'rating_norm': 0.4,'image_count_norm':0.1,
features_norm = [f+'_norm' for f in features]
df[features_norm] = (df[features]-mean_series)/range_series
df['rank_score'] = (df[features_norm]*weight_series).sum(axis=1)

grouped = df.groupby('product_id')

with open('/Users/hochen/Documents/Projects/prd-review/717283066_topic_sort.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(df.columns.values))
    writer.writeheader()

    for name, group in grouped:
        group = group.sort_values('rank_score', ascending=False)
        for i in range(group.shape[0]):
            writer.writerow(group.iloc[i].to_dict())

