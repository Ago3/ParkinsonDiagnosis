#!/usr/bin/env python3

#Load feature values for Hold Time Task
#h_data = np.genfromtxt(filename, delimiter="\t", usecols=(3,4,5,6,7,8,9,10,11,12,13,14,15))

import os
import numpy as np
import pickle
from scipy.stats import kurtosis, skew

CLEANED_DATA_DIR = "./cleaned_data/"
AGGR_DATA_DIR = "./aggregated_data/"
TRUE_CLASSES = "users_diagnosis.pkl"

def aggregate_datafile(filename, hasPD):
	user = filename.split("/")[2].split("_")[0]
	data = np.genfromtxt(filename, delimiter="\t", usecols=(3,4,5,6,7,8,9,10,11,12,13,14,15,16,17))
	#Left finger : mean, standard deviation, skewness and kurtosis of Hold Time
	#Right finger : mean, standard deviation, skewness and kurtosis of Hold Time
	#Mean(hold_time_left) - Mean(hold_time_right)
	line = ""
	for i in range(len(data[0])):
		line += str(data[0][i]) + "\t"
	line += str(hasPD[user]) + "\n"
	return line

def aggregate_data(hasPD):
	with open(AGGR_DATA_DIR + "data.csv", "w+") as out:
		for filename in os.listdir(CLEANED_DATA_DIR):
			if os.path.isdir(CLEANED_DATA_DIR + filename) or filename[-3:] == "pkl":
				print("Skipping: " + CLEANED_DATA_DIR + filename + "\n")
				continue
			else:
				out.write(aggregate_datafile(CLEANED_DATA_DIR + filename, hasPD))
				break


with open(CLEANED_DATA_DIR + TRUE_CLASSES, "rb") as dump:
	hasPD = pickle.load(dump)
aggregate_data(hasPD)