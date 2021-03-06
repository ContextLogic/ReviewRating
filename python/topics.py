import argparse
import csv
import json
import operator
import os
import pickle
import re

import pandas as pd
from gensim import matutils
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords

import utils

MODEL_FILE = 'models/731458419_model.pkl'
CORRELATION_GATE = 0
CLUSTER_MODEL = 'models/731458419_word_cluster.pkl'
CLUSTER_LABEL = 'knowledge/731458419_word_cluster.csv'


class TopicAnalyzer:

    def __init__(self):

        print(os.getcwd())
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        self.model_ak = pickle.load(open(MODEL_FILE, 'rb'))
        self.cluster = pickle.load(open(CLUSTER_MODEL, 'rb'))
        means = self.cluster.means()
        for i in range(len(means)):
            means[i] = matutils.unitvec(means[i])

        self.cluster_label = {}
        with open(CLUSTER_LABEL,'r') as f:
            csv_reader = csv.reader(f)
            for k, row in enumerate(csv_reader):
                if len(row[1]) > 0:
                    self.cluster_label[k] = row[1]

        with open('knowledge/topics2.json', 'r') as fp:
            self.topic_keys = json.load(fp)

        self.sentiment_keys = {
            'positive_sentiment': ['excellent'],
            'negative_sentiment': ['horrible']}

    def run(self, input_file, rows, category):

        self.topic_keys = self.topic_keys[category]
        filename, file_extension = os.path.splitext(input_file)

        output_file = filename + '_' + category.replace(" ", "") + '_topic' + file_extension
        df = utils.loadDataFromCsv(input_file, rows=rows)

        df = df[df['parent_tag_name'] == category]
        df['comment'] = df['comment'].fillna('')

        topic_column = df.apply(self.topicAnalysis_cluster, axis=1)
        # topic_column = df.apply(self.topicAnalysis, axis=1)
        # sentiment_column = df.apply(self.sentimentAnalysis, axis=1)

        df = pd.concat(
            [df[['product_id', 'transaction_id', 'rating_id', 'rating', 'comment', 'image_count', 'true_tag_name',
                 'sub_tag_name']], topic_column],
            axis=1)  # , sentiment_column

        """        features = ['word_count', 'topic_count']

        mean_series = df[features].mean()
        print("mean_series: " + str(mean_series))
        max_series = df[features].max()
        print("max_series: " + str(max_series))
        min_series = df[features].min()
        print("min_series: " + str(min_series))

        range_series = pd.DataFrame([max_series - mean_series, mean_series - min_series]).max()
        weight_series = pd.Series(
            {'word_count_norm': 0.1, 'topic_count_norm': 0.5})  # 'rating_norm': 0.4,'image_count_norm':0.1,
        features_norm = [f + '_norm' for f in features]
        df[features_norm] = (df[features] - mean_series) / range_series
        df['rank_score'] = (df[features_norm] * weight_series).sum(axis=1)"""
        df['rank_score'] = 0

        grouped = df.groupby('product_id')

        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(df.columns.values))
            writer.writeheader()

            for name, group in grouped:
                group = group.sort_values('rank_score', ascending=False)
                for i in range(group.shape[0]):
                    writer.writerow(group.iloc[i].to_dict())

    def preprocess_text(self, text):
        text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
        text = re.sub(' +', ' ', text)
        return text.strip()

    def topicAnalysis(self, row):
        try:
            scores = {}
            word_count = 0
            for sentence in nltk.sent_tokenize(row['comment']):
                sentence = self.preprocess_text(sentence.lower())
                words = sentence.split(' ')
                words = list(filter(lambda w: w not in self.stop_words, words))
                word_count += len(words)

                topic_score_dict = {}
                for topic, seeds in self.topic_keys.items():
                    topic_score = 0
                    for word in words:
                        score = max(
                            [(self.model_ak.wv.similarity(word, seed) if word in self.model_ak.wv.vocab else 0) for seed
                             in
                             seeds])
                        topic_score += (score if score > CORRELATION_GATE else 0)

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

            series = pd.Series({'word_count': word_count, 'topic_count': len(topic_list), 'topic_list': topic_list,
                                'topic_score': f"{scores}"})  # word_count, topic_list, f"{scores}"

            return series
        except:
            print(row['product_id'], row['transaction_id'])
            print(row['comment'])
            series = pd.Series({'word_count': -1, 'topic_count': 0, 'topic_list': None,
                                'topic_score': ""})  # word_count, topic_list, f"{scores}"

            return series

    def topicAnalysis_cluster(self, row):
        try:
            scores = {}
            word_count = 0
            for sentence in nltk.sent_tokenize(row['comment']):
                sentence = self.preprocess_text(sentence.lower())
                words = sentence.split(' ')
                words = list(filter(lambda w: w not in self.stop_words, words))
                word_count += len(words)

                topic_score_dict = {}
                for value in self.cluster_label.values():
                    topic_score_dict[value]=0

                for word in words:
                    if word in self.model_ak.wv.vocab:
                        score_list = [matutils.unitvec(self.model_ak.wv[word]).dot(v) for v in self.cluster.means()]
                        score = max(score_list)
                        index = score_list.index(score)
                        if index in self.cluster_label and score > CORRELATION_GATE:
                            topic_score_dict[self.cluster_label[index]] += score

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

            series = pd.Series({'word_count': word_count, 'topic_count': len(topic_list), 'topic_list': topic_list,
                                'topic_score': f"{scores}"})  # word_count, topic_list, f"{scores}"

            return series
        except:
            print(row['product_id'], row['transaction_id'])
            print(row['comment'])
            series = pd.Series({'word_count': -1, 'topic_count': 0, 'topic_list': None,
                                'topic_score': ""})  # word_count, topic_list, f"{scores}"

            return series

    def sentimentAnalysis(self, row):
        scores = {}
        sentiment_score_dict = {}

        for sentence in nltk.sent_tokenize(row['comment']):
            sentence = self.preprocess_text(sentence.lower())
            words = sentence.split(' ')

            for sentiment, seeds in self.sentiment_keys.items():
                sentiment_score = 0
                for word in words:
                    score = max(
                        [(self.model_ak.wv.similarity(word, seed) if word in self.model_ak.wv.vocab else 0) for seed in
                         seeds])
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

        return pd.Series({'sentiment': scores})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Topic of Product Reviews')
    parser.add_argument('-f', '--input_file', metavar='INPUT_FILE', type=str,
                        help='input file path')
    parser.add_argument('-n', '--rows', metavar='ROWS', type=int, default=-1,
                        help='rows to process')
    parser.add_argument('-c', '--category', metavar='CATEGORY', type=str, default="Home Decor",
                        help='category to process')

    parser.print_help()
    args = parser.parse_args()

    ta = TopicAnalyzer()
    ta.run(args.input_file, args.rows, args.category)
