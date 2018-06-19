#!/usr/bin/env python3

import os
import re
from datetime import datetime
import time
import ast
import pickle

DATA_DIR = "./Tappy Data/"
USER_DIR = "./Archived users/"
CLEANED_DATA_DIR = "./cleaned_data/"
num_instances = 0

def clean_data():
    for filename in os.listdir(DATA_DIR):
        if os.path.isdir(DATA_DIR + filename):
            print("Skipping: " + DATA_DIR + filename + "\n")
            continue
        with open(DATA_DIR + filename, "r") as f:
            with open(CLEANED_DATA_DIR + filename[0:-3] + "csv", "w+") as out:
                print("Reading: " + DATA_DIR + filename + "\n")
                print("Writing: " + CLEANED_DATA_DIR + filename[0:-3] + "csv" + "\n")
                lines = f.readlines()
                for ln, line in enumerate(lines):
                    try:
                        words = line.split()

                        if (len(words[1]) != 6):
                            continue
                        if (len(words[2]) != 12):
                            continue
                        if (len(words[3]) != 1 or (words[3] != "L" and words[3] != "S" and words[3] != "R")):
                            continue
                        if (len(words[5]) != 2 or re.search("[^rslRSL]", words[5])):
                            continue
                        
                        #Convert timestamps to integers
                        dt_obj = datetime.strptime(words[2], "%H:%M:%S.%f")
                        timestamp = int(time.mktime(dt_obj.timetuple()))
                        line = line.replace(words[2], str(timestamp))
                        
                        #Replace Hand feature by binary features
                        if (words[3] == "L"):
                            line = line.replace("\t" + words[3] + "\t", "\t1\t0\t0\t")
                        elif (words[3] == "R"):
                            line = line.replace("\t" + words[3] + "\t", "\t0\t1\t0\t")
                        elif (words[3] == "S"):
                            line = line.replace("\t" + words[3] + "\t", "\t0\t0\t1\t")

                        #Replace Direction feature by binary features
                        if(words[5] == "LL"):
                            line = line.replace("\t" + words[5] + "\t", "\t1\t0\t0\t0\t0\t0\t0\t0\t0\t")
                        elif(words[5] == "LR"):
                            line = line.replace("\t" + words[5] + "\t", "\t0\t1\t0\t0\t0\t0\t0\t0\t0\t")
                        elif(words[5] == "LS"):
                            line = line.replace("\t" + words[5] + "\t", "\t0\t0\t1\t0\t0\t0\t0\t0\t0\t")
                        elif(words[5] == "RL"):
                            line = line.replace("\t" + words[5] + "\t", "\t0\t0\t0\t1\t0\t0\t0\t0\t0\t")
                        elif(words[5] == "RR"):
                            line = line.replace("\t" + words[5] + "\t", "\t0\t0\t0\t0\t1\t0\t0\t0\t0\t")
                        elif(words[5] == "RS"):
                            line = line.replace("\t" + words[5] + "\t", "\t0\t0\t0\t0\t0\t1\t0\t0\t0\t")
                        elif(words[5] == "SL"):
                            line = line.replace("\t" + words[5] + "\t", "\t0\t0\t0\t0\t0\t0\t1\t0\t0\t")
                        elif(words[5] == "SR"):
                            line = line.replace("\t" + words[5] + "\t", "\t0\t0\t0\t0\t0\t0\t1\t0\t0\t")
                        elif(words[5] == "SS"):
                            line = line.replace("\t" + words[5] + "\t", "\t0\t0\t0\t0\t0\t0\t0\t0\t1\t")
                        
                        try:
                            float(words[4])
                            float(words[6])
                            float(words[7])
                        except ValueError:
                            continue
                        out.write(line)
                        num_instances += 1
                    except IndexError:
                        print(line)

    print("Num instances: " + str(num_instances) + "\n")

def gen_trueClass():
    hasPD = dict()
    for filename in os.listdir(CLEANED_DATA_DIR):
        if os.path.isdir(CLEANED_DATA_DIR + filename) or filename[-3:] == "pkl":
            print("Skipping: " + CLEANED_DATA_DIR + filename + "\n")
            continue
        user = filename.split("_")[0]
        if user in hasPD:
            continue
        with open(USER_DIR + "User_" + user +".txt", "r") as info:
            info.readline()
            info.readline()
            diagnosis = info.readline()
            words = diagnosis.split()
            hasPD[user] = float(ast.literal_eval(words[1]))
    with open(CLEANED_DATA_DIR + "users_diagnosis.pkl", "wb") as out:
        pickle.dump(hasPD, out)

clean_data()
gen_trueClass()