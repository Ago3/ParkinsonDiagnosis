#!/usr/bin/env python3

import numpy as np
import math
from sklearn.preprocessing import Imputer, normalize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.utils import shuffle

AGGR_DATA = "./aggregated_data/data.csv"

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


h_train, l_train, y_train, h_eval, l_eval, y_eval = load_data()