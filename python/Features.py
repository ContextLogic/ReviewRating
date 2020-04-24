import csv
import logging
import pickle
import sys
from datetime import date

import pandas as pd
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import xgboost as xgb
from xgboost import XGBRegressor
from nltk import FreqDist, WordNetLemmatizer
import spacy



logger = get_logger('review rating')

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')


TRAIN_NEW = False
if TRAIN_NEW:

    df = loadDataFromCsv('/Users/hochen/Documents/Projects/prd-review/717283066_sort.csv', rows=100000)

    logger.info('start feature engineering')

    wordvec_columns = []
    for index, row in df.iterrows():
        doc = nlp(row['comment'])
        wordvec_row = [len(doc)] + doc.vector.tolist()
        wordvec_columns.append(wordvec_row)

    """    words = word_tokenize(row['comment'])
        processed_words = [word.lower() for word in words if word.isalnum() and word not in stop_words]
        freq = FreqDist(processed_words)
        words_columns.append([processed_words])"""

    wordvec_names = ['word_count'] + [f"dim_{i}" for i in range(len(wordvec_columns[0]) - 1)]

    df = pd.concat([df, pd.DataFrame(wordvec_columns, index=df.index, columns=wordvec_names)], axis=1)

    logger.info('start training')

    X_cols = ['image_count'] + wordvec_names  # , 'parent_tag_name'
    y_col = 'vote_norm'




    train_products = list(range(int(df.shape[0] * 0.8)))  # df.sample(random_state=45, frac=0.8).index
    test_products = df.index.difference(train_products)

    train_y = df.iloc[train_products][y_col].copy()
    test_y = df.iloc[test_products][y_col].copy()

    train_X = df.iloc[train_products].drop(y_col, axis=1)
    test_X = df.iloc[test_products].drop(y_col, axis=1)

    eval_set = [(train_X[X_cols], train_y), (test_X[X_cols], test_y)]

    params = {
        'n_estimators': 500,
        'max_depth': 3,
        'learning_rate': 0.01,
        'objective': "reg:squarederror"
    }

    xg = XGBRegressor(**params)

    xg = xg.fit(train_X[X_cols], train_y, eval_set=eval_set, verbose=True)

    pickle.dump(xg, open('/Users/hochen/Documents/Projects/prd-review/model.pkl', 'wb'))
    pickle.dump([train_X, train_y, test_X, test_y, X_cols, y_col],
                open('/Users/hochen/Documents/Projects/prd-review/data.pkl', 'wb'))

else:
    xg = pickle.load(open('/Users/hochen/Documents/Projects/prd-review/model.pkl', 'rb'))
    [train_X, train_y, test_X, test_y, X_cols, y_col] = pickle.load(
        open('/Users/hochen/Documents/Projects/prd-review/data.pkl', 'rb'))

    pred_y = xg.predict(test_X[X_cols])

    output = pd.concat([test_X.drop(columns=[f"dim_{i}" for i in range(384)]),
                        test_y,
                        pd.DataFrame(pred_y, index=test_X.index, columns=['pred_y'])], axis=1)

    grouped = output.groupby('product_id')
    with open('/Users/hochen/Documents/Projects/prd-review/717283066_pred.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(output.columns.values))
        writer.writeheader()

        for name, group in grouped:
            group = group.sort_values('pred_y', ascending=False)
            for i in range(group.shape[0]):
                writer.writerow(group.iloc[i].to_dict())



