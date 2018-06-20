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
	left_data = data[data[:, 0] == 1]
	right_data = data[data[:, 1] == 1]
	#space_data = data[data[:, 2] == 1]
	
	#Collect HoldTime features
	#Left finger : mean, standard deviation, skewness and kurtosis of Hold Time
	#Right finger : mean, standard deviation, skewness and kurtosis of Hold Time
	#Mean difference between Left and Right
	mean_left = np.mean(left_data, axis=0)[3]
	mean_right = np.mean(right_data, axis=0)[3]
	std_left = np.std(left_data, axis=0)[3]
	std_right = np.std(right_data, axis=0)[3]
	skewness_left = skew(left_data, axis=0)[3]
	skewness_right = skew(right_data, axis=0)[3]
	kurtosis_left = kurtosis(left_data, axis=0)[3]
	kurtosis_right = kurtosis(right_data, axis=0)[3]
	mean_difference = abs(mean_left - mean_right)

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