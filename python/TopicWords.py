from python.utils import loadDataFromCsv
import numpy
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def nonemptyTopic(row, topic_cols):
    topic_result = [row[col] if isinstance(row[col], str) else '' for col in topic_cols]

    topics = ''.join(topic_result)

    if len(topics) > 0:
        return True
    else:
        return False


def preprocess_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
    text = re.sub(' +', ' ', text)
    return text.strip()


def topic_words_count(topic_df):
    word_freq = {}
    for comment in topic_df['comment']:

        comment = preprocess_text(comment)
        word_list = nltk.word_tokenize(comment)

        for word in word_list:
            if word in stop_words:
                continue

            if word not in word_freq:
                word_freq[word] = 1
            else:
                word_freq[word] += 1

    sort_words = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)
    for x in range(20):
        print(sort_words[x])


stop_words = set(stopwords.words('english'))

input_file = "/Users/hochen/Documents/Projects/prd-review/737684276_Gadgets_topic_filtered - 737684276_Gadgets_topic_filtered.csv"
rows = -1

df = loadDataFromCsv(input_file, rows=rows)
topic_cols = [col for col in df.columns if 'topic:' in col]

df = df[df.apply(nonemptyTopic, args=(topic_cols,), axis=1)]

for col in topic_cols:

    if col == 'topic: other (pls. specify)':
        continue

    print(col)

    topic_df = df[df.apply(lambda row: len(row[col]) > 0 if isinstance(row[col], str) else False, axis=1)]

    topic_words_count(topic_df)

col == 'topic: other (pls. specify)'
other_topics = {}
for t in df[col]:
    if not isinstance(t, str):
        continue

    t_list = t.split('/')
    for tmp in t_list:
        if tmp not in other_topics:
            other_topics[tmp] = 1
        else:
            other_topics[tmp] += 1

print(other_topics)

for t in other_topics:
    if other_topics[t] < 5:
        continue
    print(t)

    topic_df = df[
        df.apply(lambda row: len(row[col]) > 0 and t in row[col] if isinstance(row[col], str) else False, axis=1)]

    topic_words_count(topic_df)