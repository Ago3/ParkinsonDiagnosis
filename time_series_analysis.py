#!/usr/bin/env python3

import os
import numpy as np
import pickle
from scipy.stats import kurtosis, skew

CLEANED_DATA_DIR = "./cleaned_data/"
AGGR_DATA_DIR = "./aggregated_data/"
TRUE_CLASSES = "users_diagnosis.pkl"
FEATURES = ["mean_L", "std_L", "skew_L", "kurt_L", "mean_R", "std_R", "skew_R", "kurt_R", "mean_diff",
	"mean_LR", "std_LR", "skew_LR", "kurt_LR", "mean_RL", "std_RL", "skew_RL", "kurt_RL", "mean_diff_LR_RL",
	"mean_LL", "std_LL", "skew_LL", "kurt_LL", "mean_RR", "std_RR", "skew_RR", "kurt_RR", "mean_diff_LL_RR"]

def aggregate_datafile(filename, hasPD):
	user = filename.split("/")[2].split("_")[0]
	data = np.genfromtxt(filename, delimiter="\t", usecols=(3,4,5,6,7,8,9,10,11,12,13,14,15,16,17))
	if (data.ndim < 2):
		print("Warning: small file\n")
		return ""
	left_data = data[data[:, 0] == 1]
	right_data = data[data[:, 1] == 1]
	#space_data = data[data[:, 2] == 1]

	left_to_right_data = data[data[:, 5] == 1]
	right_to_left_data = data[data[:, 7] == 1]
	left_to_left_data = data[data[:, 4] == 1]
	right_to_right_data = data[data[:, 8] == 1]
	
	#Feature Group: Hold
	#Left finger : mean, standard deviation, skewness and kurtosis of Hold Time
	#Right finger : mean, standard deviation, skewness and kurtosis of Hold Time
	#Mean difference between Left and Right
	if len(left_data) > 0:
		mean_left = np.mean(left_data, axis=0)[3]
		std_left = np.std(left_data, axis=0)[3]
		skewness_left = skew(left_data, axis=0)[3]
		kurtosis_left = kurtosis(left_data, axis=0)[3]
	else:
		mean_left = ""
		std_left = ""
		skewness_left = ""
		kurtosis_left = ""
	if len(right_data) > 0:
		std_right = np.std(right_data, axis=0)[3]
		mean_right = np.mean(right_data, axis=0)[3]
		skewness_right = skew(right_data, axis=0)[3]
		kurtosis_right = kurtosis(right_data, axis=0)[3]
	else:
		mean_right = ""
		std_right = ""
		skewness_right = ""
		kurtosis_right = ""
	if len(right_data) > 0 and len(left_data) > 0:
		mean_difference = abs(mean_left - mean_right)
	else:
		mean_difference = ""

	#Feature Group: Latency
	#Left to Right: mean, standard deviation, skewness and kurtosis of Latency Time
	#Right to Left: mean, standard deviation, skewness and kurtosis of Latency Time
	#Left to Left: mean, standard deviation, skewness and kurtosis of Latency Time
	#Right to Right: mean, standard deviation, skewness and kurtosis of Latency Time
	#Mean difference between LR and RL
	#Mean difference between LL and RR
	if len(left_to_right_data) > 0:
		mean_left_to_right = np.mean(left_to_right_data, axis=0)[13]
		std_left_to_right = np.std(left_to_right_data, axis=0)[13]
		skewness_left_to_right = skew(left_to_right_data, axis=0)[13]
		kurtosis_left_to_right = kurtosis(left_to_right_data, axis=0)[13]
	else:
		mean_left_to_right = ""
		std_left_to_right = ""
		skewness_left_to_right = ""
		kurtosis_left_to_right = ""
	if len(right_to_left_data) > 0:
		mean_right_to_left = np.mean(right_to_left_data, axis=0)[13]
		std_right_to_left = np.std(right_to_left_data, axis=0)[13]
		skewness_right_to_left = skew(right_to_left_data, axis=0)[13]
		kurtosis_right_to_left = kurtosis(right_to_left_data, axis=0)[13]
	else:
		mean_right_to_left = ""
		std_right_to_left = ""
		skewness_right_to_left = ""
		kurtosis_right_to_left = ""
	if len(left_to_right_data) > 0 and len(right_to_left_data) > 0:
		mean_difference_LR_RL = abs(mean_left_to_right - mean_right_to_left)
	else:
		mean_difference_LR_RL = ""

	if len(left_to_left_data) > 0:
		mean_left_to_left = np.mean(left_to_left_data, axis=0)[13]
		std_left_to_left = np.std(left_to_left_data, axis=0)[13]
		skewness_left_to_left = skew(left_to_left_data, axis=0)[13]
		kurtosis_left_to_left = kurtosis(left_to_left_data, axis=0)[13]
	else:
		mean_left_to_left = ""
		std_left_to_left = ""
		skewness_left_to_left = ""
		kurtosis_left_to_left = ""
	if len(right_to_right_data) > 0:
		mean_right_to_right = np.mean(right_to_right_data, axis=0)[13]
		std_right_to_right = np.std(right_to_right_data, axis=0)[13]
		skewness_right_to_right = skew(right_to_right_data, axis=0)[13]
		kurtosis_right_to_right = kurtosis(right_to_right_data, axis=0)[13]
	else:
		mean_right_to_right = ""
		std_right_to_right = ""
		skewness_right_to_right = ""
		kurtosis_right_to_right = ""
	if len(left_to_left_data) > 0 and len(right_to_right_data) > 0:
		mean_difference_LL_RR = abs(mean_left_to_left - mean_right_to_right)
	else:
		mean_difference_LL_RR = ""

	line = str(mean_left) + "\t" + str(std_left) + "\t" + str(skewness_left) + "\t" + str(kurtosis_left) + "\t"
	line += str(mean_right) + "\t" + str(std_right) + "\t" + str(skewness_right) + "\t" + str(kurtosis_right) + "\t" + str(mean_difference) + "\t"
	line += str(mean_left_to_right) + "\t" + str(std_left_to_right) + "\t" + str(skewness_left_to_right) + "\t" + str(kurtosis_left_to_right) + "\t"
	line += str(mean_right_to_left) + "\t" + str(std_right_to_left) + "\t" + str(skewness_right_to_left) + "\t" + str(kurtosis_right_to_left) + "\t" + str(mean_difference_LR_RL) + "\t"
	line += str(mean_left_to_left) + "\t" + str(std_left_to_left) + "\t" + str(skewness_left_to_left) + "\t" + str(kurtosis_left_to_left) + "\t"
	line += str(mean_right_to_right) + "\t" + str(std_right_to_right) + "\t" + str(skewness_right_to_right) + "\t" + str(kurtosis_right_to_right) + "\t" + str(mean_difference_LL_RR) + "\t" 
	line += str(hasPD[user]) + "\n"
	return line

def aggregate_data(hasPD):
	with open(AGGR_DATA_DIR + "data.csv", "w+") as out:
		header = ""
		for feature in FEATURES:
			header += feature + "\t"
		header += "hasPD" + "\n"
		out.write(header)
		for filename in os.listdir(CLEANED_DATA_DIR):
			if os.path.isdir(CLEANED_DATA_DIR + filename) or filename[-3:] == "pkl":
				print("Skipping: " + CLEANED_DATA_DIR + filename + "\n")
				continue
			else:
				#print("Working on: " + CLEANED_DATA_DIR + filename + "\n")
				line = aggregate_datafile(CLEANED_DATA_DIR + filename, hasPD)
				if line != "":
					out.write(line)

with open(CLEANED_DATA_DIR + TRUE_CLASSES, "rb") as dump:
	hasPD = pickle.load(dump)
aggregate_data(hasPD)