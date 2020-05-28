import csv
import pickle
from gensim.models import Word2Vec
from nltk.cluster import KMeansClusterer
import nltk

MODEL_FILE = 'models/731458419_model.pkl'#'models/737684276_Gadgets_topic_model.pkl'#
model_ak: Word2Vec = pickle.load(open(MODEL_FILE, 'rb'))

"""input_file = "/Users/hochen/Documents/Projects/prd-review/737684276_Gadgets_topic_filtered - 737684276_Gadgets_topic_filtered.csv"
rows = -1
df = loadDataFromCsv(input_file, rows=rows)"""

MODEL_FILE2 = 'models/737684276_Gadgets_topic_model.pkl'#
model_ak2: Word2Vec = pickle.load(open(MODEL_FILE2, 'rb'))

word_list:list = list(model_ak2.wv.vocab)[:15000]

X=[]
for w in word_list:
    if w in model_ak.wv.vocab:
        X.append(model_ak[w])

NUM_CLUSTERS=350
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
print("start clustering ...")
assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
#print (assigned_clusters)

pickle.dump(kclusterer, open("models/737684276_Gadgets_topic_cluster.pkl", 'wb'))

with open("models/737684276_Gadgets_topic_word_cluster.csv", 'w') as f:
    for i in range(NUM_CLUSTERS):
        writer = csv.writer(f)

        line = []
        for t in model_ak.wv.similar_by_vector(kclusterer.means()[i], topn=10, restrict_vocab=15000):
            if t[0] in word_list:
                index = word_list.index(t[0])
                if assigned_clusters[index] == i:
                    line.append(f"{t[0]}:{t[1]}")
                else:
                    print("cross boundary")

        writer.writerow([";".join(line), ""])
