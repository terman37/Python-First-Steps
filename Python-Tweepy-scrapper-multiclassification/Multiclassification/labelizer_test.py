from ast import literal_eval
import re

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.feature_extraction import stop_words
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier

# A host of Scikit-learn models
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

SEED = 222

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stop_words.ENGLISH_STOP_WORDS)


def preprocess(text):
    """
        text: a string
        return: modified initial string
    """
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = " ".join([token for token in text.split() if token not in STOPWORDS])  # delete stopwords from text
    return text

def read_data2(filename):
    data = pd.read_csv(filename, sep=';')
    indexNames = data[data['title'] == 'ERROR'].index
    data.drop(indexNames, inplace=True)
    # data['tags'] = data['tags'].apply(literal_eval)  # safely evaluate a string expression
    data['title'] = data['title'].apply(preprocess)
    return data

def read_data(filename):
    data = pd.read_csv(filename, sep=';')
    data['tags'] = data['tags'].apply(literal_eval)  # safely evaluate a string expression
    data['title'] = data['title'].apply(preprocess)
    return data


tweets = read_data2('data/tweets.csv')
print(tweets.shape)

X_train, X_test, y_train, y_test = train_test_split(tweets['title'], tweets['tags'], test_size=0.25, random_state=42)

# # train = read_data('data/train.tsv')
# # validation = read_data('data/validation.tsv')
# #
# # type(train)
# # train[2] = train['tags'].apply(tuple)
# # train.groupby(2).groups
# # validation[2] = validation['tags'].apply(tuple)
#
# ################  MAKE TRAIN
#
# YT_unique_tags = set([y for y in y_train])
# tag_counts = [len(e) for e in y_train]
# le_tr = preprocessing.LabelEncoder()
# le_tr.fit(sorted(list(YT_unique_tags)))
# le_tr.classes_
#
# # YT_unique_tags = set([t for y in train['tags'] for t in y])
# # tag_counts = [len(e) for e in train['tags']]
# # le_tr = preprocessing.LabelEncoder()
# # le_tr.fit(sorted(list(YT_unique_tags)))
# # le_tr.classes_
#
# YV_unique_tags = set([y for y in y_test])
# # test intersection
#
# # Encode the labels
# le_train = y_train.apply(lambda x: le_tr.transform(x))
#
# ######################  MANUEL PAD
#
# rtr = int(np.amax(train['le'].apply(lambda x: np.count_nonzero(x))))
# ctr = train['le'].size
# atr = train['le'].values
#
# ytrain = np.zeros((ctr, rtr))
# for i, j in enumerate(atr):
#     ytrain[i][0:len(j)] = j
#
# # Inverse the encoding
# train['le'].apply(lambda x: le_tr.inverse_transform(x))
#
# ###################  MAKE YTEST
#
# ################  MAKE TRAIN
#
# YV_unique_tags = set([t for y in validation['tags'] for t in y])
# tag_counts = [len(e) for e in validation['tags']]
# le_te = preprocessing.LabelEncoder()
# le_te.fit(sorted(list(YT_unique_tags)))
# le_te.classes_
#
# # test intersection
#
# # Encode the labels
# validation['le'] = validation['tags'].apply(lambda x: le_te.transform(x))
#
# ######################  MANUEL PAD
#
# rte = int(np.amax(validation['le'].apply(lambda x: np.count_nonzero(x))))
# cte = validation['le'].size
# ate = validation['le'].values
#
# ytest = np.zeros((cte, rte))
# for i, j in enumerate(ate):
#     ytest[i][0:len(j)] = j
#
# # Inverse the encoding
# validation['le'].apply(lambda x: le_te.inverse_transform(x))

import spacy

# nlp = spacy.load('en')
# nlp.vocab.vectors.from_glove('/path/to/vectors')

# TODO - no effect python -m spacy download en_core_web_lg for big trained vectors
# python -m spacy download en_core_web_lg for smaller trained vectors

# TODO me! python -m spacy download en_core_web_md for big trained vectors
nlp = spacy.load('en_core_web_md')
#####need a workaround 11/2018 due to msgpack dependencies
# python -m pip install "msgpack<0.6.0"

train_corpus = X_train
#train_corpus = train['title'].tolist()
tokens = [nlp(x) for x in train_corpus]
# check entries

print(len(tokens))

text = []
vec = []
norm = []
tensor = []
sent = []
for token in tokens:
    # print(token.text, token.has_vector, token.vector_norm, token.is_oov)
    text.append(token.text)
    vec.append(token.vector)
    norm.append(token.vector_norm)
    tensor.append(token.tensor)
    sent.append(token.sents)

len(tensor)
xtrain = np.asarray(vec)

#############################  MAKE XTEST
#test_corpus = validation['title'].tolist()
test_corpus = X_test
tokensTe = [nlp(x) for x in test_corpus]
# check entries

print(len(tokensTe))

textTe = []
vecTe = []
normTe = []
tensorTe = []
sentTe = []
for token in tokensTe:
    # print(token.text, token.has_vector, token.vector_norm, token.is_oov)
    textTe.append(token.text)
    vecTe.append(token.vector)
    normTe.append(token.vector_norm)
    tensorTe.append(token.tensor)
    sentTe.append(token.sents)

len(tensorTe)
xtest = np.asarray(vecTe)

print(type(xtrain))
print(type(y_train))
print(type(xtest))
print(type(y_test))

# print(y_train[1])
# print(y_test[1])

# save pickle file
import pickle

pickle.dump(xtrain, open('data/xtrain.pkl', 'wb'))
pickle.dump(y_train, open('data/ytrain.pkl', 'wb'))
pickle.dump(xtest, open('data/xtest.pkl', 'wb'))
pickle.dump(y_test, open('data/ytest.pkl', 'wb'))
# load pickle file
ytrain = pickle.load(open('data/ytrain.pkl', 'rb'))
xtrain = pickle.load(open('data/xtrain.pkl', 'rb'))
xtest = pickle.load(open('data/xtest.pkl', 'rb'))
ytest = pickle.load(open('data/ytest.pkl', 'rb'))

# import pydotplus  # you can install pydotplus with: pip install pydotplus
# from IPython.display import Image
from sklearn.metrics import roc_auc_score

def get_models():
    """Generate a library of base learners."""
    nb = GaussianNB()
    svc = SVC(C=100, probability=True)
    knn = KNeighborsClassifier(n_neighbors=3)
    lr = LogisticRegression(C=100, random_state=SEED)
    nn = MLPClassifier((80, 10), early_stopping=False, random_state=SEED)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=SEED)
    rf = RandomForestClassifier(n_estimators=10, max_features=3, random_state=SEED)
    adb = AdaBoostClassifier()

    # qda = OneVsRestClassifier(QuadraticDiscriminantAnalysis(priors=None, reg_param=0.001,
    #                          store_covariance=True,
    #                         tol=0.0001))

    models = {'svm': svc,
              'knn': knn,
              'naive bayes': nb,
              'mlp-nn': nn,
              'random forest': rf,
              'gbm': gb,
              'logistic': lr,
              'adaboost': adb,
              # 'quadratic discriminant' : qda
              }

    return models


def train_predict(model_list):
    """Fit models in list on training set and return predictions"""
    P = np.zeros((ytest.shape[0], len(model_list)))
    P = pd.DataFrame(P)

    print("Fitting models.")
    cols = list()
    for i, (name, m) in enumerate(models.items()):
        print("%s..." % name, end=" ", flush=False)
        n = MultiOutputClassifier(m)
        n.fit(xtrain, ytrain)
        P.iloc[:, i] = n.predict(xtest)
        cols.append(name)
        print("done")

    P.columns = cols
    print("Done.\n")
    return P


def score_models(P, y):
    """Score model in prediction DF"""
    print("Scoring models.")
    for m in P.columns:
        score = roc_auc_score(y, P.loc[:, m])
        print("%-26s: %.3f" % (m, score))
    print("Done.\n")


models = get_models()
P = train_predict(models)
score_models(P, ytest)

# You need ML-Ensemble for this figure: you can install it with: pip install mlens


corrmat(P.corr(), inflate=False)
plt.show()

# n_classes = 3
forest = RandomForestClassifier(n_estimators=10, random_state=SEED, max_features=len(YT_unique_tags))
multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
p_forest = multi_target_forest.fit(xtrain, ytrain).predict(xtest)

svc = SVC(C=1, gamma='auto', probability=True)
multi_target_svc = MultiOutputClassifier(svc, n_jobs=-1)
p_svc = multi_target_svc.fit(xtrain, ytrain).predict(xtest)
