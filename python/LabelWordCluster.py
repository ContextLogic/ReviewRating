import csv
import pickle

CLUSTER_MODEL = 'models/731458419_word_cluster.pkl'
cluster = pickle.load(open(CLUSTER_MODEL, 'rb'))

MODEL_FILE = 'models/731458419_model.pkl'
model_ak = pickle.load(open(MODEL_FILE, 'rb'))


with open("models/731458419_word_cluster.csv", 'w') as f:
    writer = csv.writer(f)

    for i in range(len(cluster.means())):
        writer = csv.writer(f)

        line = []
        for k, t in enumerate(model_ak.wv.similar_by_vector(cluster.means()[i], topn=50, restrict_vocab=15000)):
            word_list = [model_ak.wv[t[0]].dot(v) for v in cluster.means()]
            p = word_list.index(max(word_list))
            if p == i:
                line.append(f"{k}: {t[0]}: {t[1]}")
            else:
                line.append(f"* {k}:  {t[0]}: {t[1]}")
                #break

        writer.writerow(["\r\n".join(line), ""])




