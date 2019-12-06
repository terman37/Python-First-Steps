from ast import literal_eval
import re

# from metrics import roc_auc, pr_isof1_plot
import numpy as np
import pandas as pd

# import seaborn as sns
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, average_precision_score
# from sklearn.metrics import f1_score, hamming_loss
# from sklearn.metrics import roc_auc_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
# from mlens.visualization import corrmat


# A host of Scikit-learn models
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
# from sklearn.kernel_approximation import Nystroem
# from sklearn.kernel_approximation import RBFSampler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputClassifier

# import pydotplus  # you can install pydotplus with: pip install pydotplus
# from IPython.display import Image
from sklearn.metrics import roc_auc_score, explained_variance_score, mean_squared_error, r2_score

# from sklearn.tree import DecisionTreeClassifier, export_graphviz

import pickle

SEED = 222
# load pickle file
ytrain = pickle.load(open('data/ytrain.pkl', 'rb'))
xtrain = pickle.load(open('data/xtrain.pkl', 'rb'))
xtest = pickle.load(open('data/xtest.pkl', 'rb'))
ytest = pickle.load(open('data/ytest.pkl', 'rb'))


def get_models():
    """Generate a library of base learners."""
    nb = GaussianNB()
    svc = SVC(C=100, gamma='auto', probability=True)
    knn = KNeighborsClassifier(n_neighbors=3)
    lr = LogisticRegression(C=100, random_state=SEED)
    nn = MLPClassifier((80, 10), early_stopping=False, random_state=SEED)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=SEED)
    rf = RandomForestClassifier(n_estimators=10, max_features=3, random_state=SEED)
    adb = AdaBoostClassifier()

    # qda = OneVsRestClassifier(QuadraticDiscriminantAnalysis(priors=None, reg_param=0.001,
    #                          store_covariance=True,
    #                         tol=0.0001))

    models = {  # 'svm': svc,
        ## 'knn': knn,
        'naive bayes': nb,
        # 'mlp-nn': nn,
        # 'random forest': rf,
        # 'gbm': gb,
        # 'logistic': lr,
        # 'adaboost': adb,
        # 'quadratic discriminant' : qda
    }

    return models


class OutputClassifier(object):
    pass


def train_predict(model_list):
    """Fit models in list on training set and return predictions"""
    P = np.zeros((ytest.shape[0], len(model_list)))
    P = pd.DataFrame(P)

    print("Fitting models.")
    cols = list()
    for i, (name, m) in enumerate(models.items()):
        print("%s..." % name, end=" ")  # , flush=False)
        # n = MultiOutputClassifier(m)
        n = OutputClassifier()
        n.fit(xtrain, ytrain)
        P.iloc[:, i] = n.predict(xtest)
        cols.append(name)
        print("done")

    P.columns = cols
    print("Done.\n")
    return P


# def score_models(P, y):
#     """Score model in prediction DF"""
#     print("Scoring models.")
#
#     ve = [explained_variance_score(y, P.loc[:, m]) for m in P.columns]
#     mse = [mean_squared_error(y, P.loc[:, m]) for m in P.columns]
#     r2 = [r2_score(y, P.loc[:, m]) for m in P.columns]
#     names = pd.Series(['ve', 'mse', 'r2'])
#     t = pd.DataFrame(data=[ve, mse, r2])
#     t.columns = P.columns
#     t['measures'] = names.values
#     print(t)
#     return t

def score_models(P, y):
    """Score model in prediction DF"""
    print("Scoring models.")
    for m in P.columns:
        score = roc_auc_score(y, P.loc[:, m], average="weighted")
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
