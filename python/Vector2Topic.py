import datetime
import math
import pickle

from python.EarlyStopping import EarlyStopping
from python.utils import loadDataFromCsv
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from gensim.models import Word2Vec
import torch




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


def feature_generation(topic_df):
    features = []
    for comment in topic_df['comment']:
        comment = preprocess_text(comment)
        word_list = nltk.word_tokenize(comment)

        features.append(word_list)

    return features


def printCentralWords(model, model_ak: Word2Vec, n):
    weight = model['input_linear.weight']
    for i in range(weight.size()[0]):
        print(f"{i}:")

        for t in model_ak.wv.similar_by_vector(weight[i].numpy(), topn=n, restrict_vocab=2000):
            print(t[0], t[1])

    highest_corr = torch.tensor(-1.0)
    for i in range(weight.size()[0] - 1):
        for j in range(i + 1, weight.size()[0]):
            corr = weight[i].dot(weight[j])
            highest_corr = torch.abs(torch.max(corr, highest_corr))

    print(f"highest correlation: {highest_corr.item()}")


def printCentralWords2(model, model_ak: Word2Vec, n):
    print("central words:")

    vectors = model['input_linear.weight']
    weights = model['output_linear.weight']

    center_vector = weights.mm(vectors)

    for t in model_ak.wv.similar_by_vector(center_vector.reshape(-1).numpy(), topn=n, restrict_vocab=2000):
        print(t[0], t[1])


def trainMapper(features, label):
    D_in = 200
    H = 5
    D_out = 1

    alpha = 10
    beta = 50

    #model = TopicNet(D_in, H, D_out)
    #model = TopicNet2(D_in, D_out)
    model = TopicNet3(D_in, H, D_out)

    label = torch.tensor(label.values)

    # criterion = torch.nn.MSELoss(reduction='sum')
    criterion = torch.nn.HingeEmbeddingLoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9, weight_decay=0.5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)
    es = EarlyStopping(patience=5)

    trainLoss = np.double(0)
    for epoch in range(300):
        # Forward pass: Compute predicted y by passing x to the model
        trainLoss = np.double(0)
        y_pred = None
        for words in features:
            score = 0
            for word in words:
                if word in model_ak.wv.vocab:
                    vector = model_ak.wv[word]
                    score += model.forward(torch.Tensor(vector))

            if y_pred is None:
                y_pred = score
            else:
                y_pred = torch.cat((y_pred, score), 0)

        # Compute and print loss
        weight = model.input_linear.weight
        highest_corr = torch.tensor(-1.0)
        for i in range(H - 1):
            for j in range(i + 1, H):
                corr = weight[i].dot(weight[j])
                highest_corr = torch.abs(torch.max(corr, highest_corr))

        """weight_loss = torch.min(model.output_linear.weight).item()
        weight_loss = math.exp(-weight_loss) - 1 if weight_loss < 0 else 0"""

        hinge_loss = criterion(y_pred.reshape(-1), label)
        loss = hinge_loss  # + alpha * highest_corr + beta * weight_loss
        trainLoss += loss.item()

        #print(f"epoch: {epoch}, train loss: {trainLoss}, hinge loss: {hinge_loss.item()}")
        print(
            f"epoch: {epoch}, train loss: {trainLoss}, hinge loss: {hinge_loss.item()}, corr loss: {highest_corr.item()}")#, weight loss: {weight_loss}")

        if es.step(loss):
            break  # early stop criterion is met, we can stop now

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step()

    print(trainLoss)
    return model


class TopicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we construct three nn.Linear instances that we will use
        in the forward pass.
        """
        super(TopicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H, bias=False)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.input_linear(x)
        y_pred = self.output_linear(h_relu).clamp(min=0)
        #y_pred = torch.sum(h_relu, dim=0, keepdim=True).clamp(min=0)
        return y_pred

class TopicNet3(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we construct three nn.Linear instances that we will use
        in the forward pass.
        """
        super(TopicNet3, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H, bias=False)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.input_linear(x)
        y_pred = h_relu.max(dim = 0, keepdim= True)[0].clamp(min=0)
        #y_pred = torch.sum(h_relu, dim=0, keepdim=True).clamp(min=0)
        return y_pred

class TopicNet2(torch.nn.Module):
    def __init__(self, D_in, D_out):
        """
        In the constructor we construct three nn.Linear instances that we will use
        in the forward pass.
        """
        super(TopicNet2, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, D_out, bias=False)

    def forward(self, x):
        y_pred = self.input_linear(x).clamp(min=0)
        return y_pred


MODEL_FILE = 'models/731458419_model.pkl'
training = False

model_ak: Word2Vec = pickle.load(open(MODEL_FILE, 'rb'))

if training:

    print(datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))

    stop_words = set(stopwords.words('english'))

    input_file = "/Users/hochen/Documents/Projects/prd-review/737684276_Gadgets_topic_filtered - 737684276_Gadgets_topic_filtered.csv"
    rows = -1

    df = loadDataFromCsv(input_file, rows=rows)
    topic_cols = [col for col in df.columns if 'topic:' in col]

    df = df[df.apply(nonemptyTopic, args=(topic_cols,), axis=1)]

    for col in topic_cols:

        if col == 'topic: other (pls. specify)':
            continue

        if col != "topic: install/instructions clear?":
            continue

        print(col)

        label = df.apply(lambda row: -1 if isinstance(row[col], str) and len(row[col]) > 0 else 1, axis=1)
        print(f"positive: {label[label == -1].count()} ; negative: {label[label == 1].count()}")

        features = feature_generation(df)

        model = trainMapper(features, label)

        torch.save(model.state_dict(),
                   'models/vector2topic.pkl' + datetime.datetime.now().strftime(".%m-%d-%Y-%H-%M-%S"))

        printCentralWords(model.state_dict(), model_ak, 10)
        # printCentralWords2(model.state_dict(), model_ak, 10)

        break

    """col = 'topic: other (pls. specify)'
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
    
        t='size'
        print(t)
    
        label = df.apply(lambda row: -1 if isinstance(row[col], str) and t in df[col] else 1, axis=1)
    
        features = feature_generation(df)
    
        model = trainMapper(features, label)
    
        torch.save(model.state_dict(),
                   'models/vector2topic.pkl' + datetime.datetime.now().strftime(".%m-%d-%Y-%H-%M-%S"))
    
        # printCentralWords(model.state_dict(), model_ak, 10)
        printCentralWords2(model.state_dict(), model_ak, 10)"""

else:

    model = torch.load('models/vector2topic.pkl.05-20-2020-12-10-19') #'models/vector2topic.pkl.05-17-2020-09-41-31'
    print()
    printCentralWords(model, model_ak, 10)
    printCentralWords2(model, model_ak, 10)
