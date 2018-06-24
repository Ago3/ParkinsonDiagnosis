#!/usr/bin/env python3

import warnings
import numpy as np
import math
from sklearn.preprocessing import Imputer, normalize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

AGGR_DATA = "./aggregated_data/data.csv"
RESULTS = "./results.csv"
Wt = 1.2

def load_data():
    #Load Hold feature group
    h_data = np.genfromtxt(AGGR_DATA, delimiter="\t", skip_header=1, usecols=(0,1,2,3,4,5,6,7,8))
    #Load Latency feature group
    l_data = np.genfromtxt(AGGR_DATA, delimiter="\t", skip_header=1, usecols=(9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26))
    #Load true classification
    y = np.genfromtxt(AGGR_DATA, delimiter="\t", skip_header=1, usecols=(27))

    if not (len(h_data) == len(l_data) and len(y) == len(h_data)):
        print("WARNING: Datasets have different lengths")

    #Replace missing values using mean imputation
    h_imp = Imputer(missing_values="NaN", strategy="mean", axis=0)
    h_imp.fit(h_data)
    h_data = h_imp.transform(h_data)
    l_imp = Imputer(missing_values="NaN", strategy="mean", axis=0)
    l_imp.fit(l_data)
    l_data = l_imp.transform(l_data)

    #Normalize values to [0,1]
    h_data = normalize(h_data, axis=0)
    l_data = normalize(l_data, axis=0)

    #Apply LDA reduction
    h_lda = LDA(solver="eigen") #, n_components=1
    h_data = h_lda.fit(h_data, y).transform(h_data)
    l_lda = LDA(solver="eigen") #, n_components=1
    l_data = l_lda.fit(l_data, y).transform(l_data)

    #Shuffle data consistently
    h_data, l_data, y = shuffle(h_data, l_data, y)

    #Reserve 20% instances for testing phase
    testing_index = int(math.ceil(len(h_data) * 0.8))
    h_train = h_data[:testing_index]
    l_train = l_data[:testing_index]
    y_train = y[:testing_index]
    h_eval = h_data[testing_index:]
    l_eval = l_data[testing_index:]
    y_eval = y[testing_index:]
    print("Training: ", len(h_train))
    print("Testing: ", len(h_eval))

    return h_train, l_train, y_train, h_eval, l_eval, y_eval


def build_classifier(x_train, y_train, mode):
    #Make 10-folds cross validation
    kf = KFold(n_splits=10, shuffle=False)
    params = dict()

    #SVM Classifier
    svm = SVC(probability=True)
    if mode == "HOLD":
        params["svm__C"] = [0.01]#list(np.logspace(-2, 8, 2))
    else:
        params["svm__C"] = [0.01]
    params["svm__gamma"] = [1000.0]#list(np.logspace(-9, 3, 2))

    #Multi-layer Perceptron
    mlp = MLPClassifier(activation="relu", batch_size=100)
    if mode == "HOLD":
        params["mlp__hidden_layer_sizes"] = [(100,)]#, (50, 50,)]
    else:
        params["mlp__hidden_layer_sizes"] = [(50, 50,)]
    #params["mlp__learning_rate_init"] = [0.001, 0.01]

    #Logistic Regression Model
    lrm = LogisticRegression()
    if mode == "HOLD":
        params["lrm__fit_intercept"] = [False]#, False]
    else:
        params["lrm__fit_intercept"] = [True]#, False]

    #Random Forest
    rf = RandomForestClassifier(criterion="entropy", n_jobs=-1)
    params["rf__n_estimators"] = [10]#, 50]

    #NSVC
    nsvc = NuSVC(nu=0.01, probability=True)
    params["nsvc__gamma"] = [0.01]#list(np.logspace(-2, 8, 2))

    #Decision Tree
    dt = DecisionTreeClassifier(criterion="entropy")

    #K-nearest Neighbors
    knn = KNeighborsClassifier()
    params["knn__n_neighbors"] = [5]#, 10]

    #Quadratic Discriminant Analysis
    qda = QuadraticDiscriminantAnalysis(tol=0.001, store_covariances=False)

    #Build ensemble
    est = [("svm", svm),
        ("mlp", mlp),
        ("lrm", lrm),
        ("rf", rf),
        ("nsvc", nsvc),
        ("dt", dt),
        ("knn", knn),
        ("qda", qda)]
    clf = VotingClassifier(est, voting='soft', n_jobs=-1)
    grid = GridSearchCV(estimator=clf, param_grid=params, cv=kf, n_jobs=-1, scoring="accuracy", verbose=1)

    print("Tuning parameters..")
    grid.fit(x_train, y_train)
    print(grid.best_params_)
    print("done!")

    return grid


def get_answers(ph, pl):
    p = (ph + 0.5 + Wt * (pl - 0.5)) / 2
    if p >= 0.5:
        return 1.0
    return 0.0


def evaluate(hold_clf, latency_clf, h_eval, l_eval, y_eval):
    print(hold_clf.score(h_eval, y_eval))
    print(confusion_matrix(y_eval, hold_clf.predict(h_eval)))
    print(latency_clf.score(l_eval, y_eval))
    print(confusion_matrix(y_eval, latency_clf.predict(l_eval)))
    ph = hold_clf.predict_proba(h_eval)[:,1]
    pl = latency_clf.predict_proba(l_eval)[:,1]
    answers = get_answers(ph, pl)
    accuracy = accuracy_score(y_eval, answers)
    print("Accuracy: ", accuracy)
    recall = recall_score(y_eval, answers)
    print("Recall: ", recall)
    f1 = f1_score(y_eval, answers)
    print("F1: ", f1)
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for index,out in enumerate(y_eval):
        if answers[index] == out and out == 1.0:
            tp = tp + 1
        elif answers[index] == out and out == 0.0:
            tn = tn + 1
        elif out == 1.0:
            fn = fn + 1
        else:
            fp = fp + 1
    print(tp, fp, tn, fn)
    with open(RESULTS, "w+") as res:
        res.write("accuracy\trecall\tf1\ttp\tfp\ttn\tfn\n")
        res.write(str(accuracy) + "\t"  + str(recall) + "\t" + str(f1) + "\t" + str(tp) + "\t" + str(fp) + "\t" + str(tn) + "\t" + str(fn) + "\n")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    h_train, l_train, y_train, h_eval, l_eval, y_eval = load_data()
    train_neg = 0
    eval_neg = 0
    for el in y_train:
        if el == 0.0:
            train_neg += 1
    for el in y_eval:
        if el == 0.0:
            eval_neg += 1
    print(train_neg, eval_neg)
    hold_clf = build_classifier(h_train, y_train, "HOLD")
    latency_clf = build_classifier(l_train, y_train, "LATENCY")
    get_answers = np.vectorize(get_answers)
    evaluate(hold_clf, latency_clf, h_eval, l_eval, y_eval)