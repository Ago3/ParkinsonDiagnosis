#!/usr/bin/env python3

import os
import re
from datetime import datetime
import time

DATA_DIR = "./Tappy Data/"
CLEANED_DATA_DIR = "./cleaned_data/"
written = 0

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
                    written += 1
                except IndexError:
                    print(line)

print(written)