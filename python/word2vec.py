import pickle
import re
import codecs
import multiprocessing
import gensim
import nltk
import spacy
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt, cm

from python.utils import loadDataFromCsv


def preprocess_text(text):
    text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
    text = re.sub(' +', ' ', text)
    return text.strip()


def prepare_for_w2v(filename_from, filename_to, lang):
    raw_text = codecs.open(filename_from, "r", encoding='windows-1251').read()
    with open(filename_to, 'w', encoding='utf-8') as f:
        for sentence in nltk.sent_tokenize(raw_text, lang):
            print(preprocess_text(sentence.lower()), file=f)


def train_word2vec(filename):
    data = gensim.models.word2vec.LineSentence(filename)
    return Word2Vec(data, size=200, window=5, min_count=3, workers=multiprocessing.cpu_count())

def tsne_plot_2d(label, embeddings, words=[], a=1):
    print('start training...')
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, 1))
    x = embeddings[:,0]
    y = embeddings[:,1]
    plt.scatter(x, y, c=colors, alpha=a, label=label)
    for i, word in enumerate(words):
        plt.annotate(word, alpha=0.3, xy=(x[i], y[i]), xytext=(5, 2),
                     textcoords='offset points', ha='right', va='bottom', size=10)
    plt.legend(loc=4)
    plt.grid(True)
    plt.savefig("hhh.png", format='png', dpi=150, bbox_inches='tight')
    plt.show()

def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.8, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    plt.show()

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')
lemmatizer = nltk.WordNetLemmatizer()

PROCESSING = False

df = loadDataFromCsv('/Users/hochen/Documents/Projects/prd-review/717283066_sort.csv', rows=-1)

train_file = '/Users/hochen/Documents/Projects/prd-review/717283066_w2v.txt'

if PROCESSING:
    with open(train_file, 'w') as f:
        for index, row in df.iterrows():
            for sentence in nltk.sent_tokenize(row['comment']):
                print(preprocess_text(sentence.lower()), file=f)

    model_ak = train_word2vec(train_file)
    pickle.dump(model_ak, open('/models/717283066_model.pkl', 'wb'))
else:
    model_ak = pickle.load(open('/models/717283066_model.pkl', 'rb'))
    #####

    keys = ['early', 'size', 'sporty', 'soft', 'color', 'concert','horrible', 'excellent']

    embedding_clusters = []
    word_clusters = []
    for word in keys:
        embeddings = [model_ak.wv[word]]
        words = [word]
        for similar_word, _ in model_ak.wv.most_similar(word, topn=30):
            words.append(similar_word+": {:.2f}".format(model_ak.wv.similarity(word, similar_word)))
            embeddings.append(model_ak.wv[similar_word])
        embedding_clusters.append(embeddings)
        word_clusters.append(words)

    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    tsne_model_en_2d = TSNE(perplexity=20, n_components=2, init='pca', n_iter=3500, random_state=32)
    embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
    tsne_plot_similar_words('Similar words from product review', keys, embeddings_en_2d, word_clusters, 0.7,
                            'similar_words.png')



    """words_ak = []
    embeddings_ak = []
    for word in list(model_ak.wv.vocab):
        embeddings_ak.append(model_ak.wv[word])
        words_ak.append(word)

    tsne_ak_2d = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3500, random_state=32)
    embeddings_ak_2d = tsne_ak_2d.fit_transform(embeddings_ak)

    tsne_plot_2d('review', embeddings_ak_2d, words_ak, 0.1)
    """

