################################################################################
# Author: Dan Wang
# May 27, 2015
# Version 0.1
# This is the command line compatible version
# Input: raw csv
# Standard Output in non-verbose mode: csv with missing value removed.
# File output: rmna_filename.csv
################################################################################

import numpy as np
import pandas as pd
import sys
import getopt
import time
pd.set_option('display.max_columns', 500)    # max columns that could be analyzed and displayed in the report: 500

def isinteger(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

# check command line header option
if len(sys.argv) == 1:
    usage = "\n##This is the help information##\n"
    usage += "Please type 'python datacheck.py file_name [-h T] [-c F] [-r F] [-v T]' to check the contents of the file\n"
    usage += "--file_name is the name of the file, including the full path and the suffix (.csv, .dat...)\n"
    usage += "--The default value of the -h option is T, meaning column headers are provided in the file.\n"
    usage += "  If the file contains no column header, use '-h F'.\n"
    usage += "--The default value of the -c option is F, meaning the distribution of the class label is not\n"
    usage += "  checked. If a column number is provided, the given column will be treated as class label, and a\n"
    usage += "  distribution will be calculated. eg: '-c 1': the first column is used as class label.\n"
    usage += "--The default value of the -r option is F, meaning missing values are not removed. If the removal\n"
    usage += "  of missing value is required, please include -r in the command line arguments.\n"
    usage += "--The -v option carries two values. True indicates verbose mode, False indicates non-verbose mode.\n"
    usage += "  Please use '-v F' if conducted in a pipeline.\n"
    exit(usage)

stats = time.strftime("%c")  # print time stamp
infile = sys.argv[1]
opts, args = getopt.getopt(sys.argv[2:], "c:h:r:v:")
header = "T"
label = "F"
verbose = "T"
rmna = "F" # default values
for o,a in opts:
    if o == '-h':
        header = a.upper()
        if header not in ['T', 'F']:
            exit("The header option is not properly provided! Please type '-h F' or '-h T'.")
    elif o == '-c':
        label = a
        if isinteger(label):
            class_col = int(label) - 1
            label = 'T'
        elif label.upper() == 'F':
                label = 'F'
        else: exit("The class option is not properly provided! Please type '-c F' or '-c #' where # is the column number.")
    elif o == '-r':
        rmna = a.upper()
        if rmna not in ["T", "F"]:
            exit("The remove missing value option is not properly provided! Please type '-r F' or '-r T'")

    elif o == '-v':
        verbose = a.upper()
        if verbose not in ["T", "F"]:
            exit("The verbose option is not properly provided! Please type '-v F' or '-v T'")


# check the number of commas in each line
with open(infile) as f:
    length = 0
    for line in f:
        line = line.split(",")
        if length == 0:
            length = len(line)
        else:
            if len(line) != length:
                exit("Unequal number of commas!")
    if length == 0:
        exit("No commas are found. This is not a .csv file.")
    else:
        stats += "\n##Summary##\nNumber of columns: "
        stats += str(length)

# read data in panda
if header == "T":
    tf = pd.read_csv(infile, iterator = True, chunksize = 10000)
    file = pd.concat(tf, ignore_index=True)
else:
    tf = pd.read_csv(infile, iterator = True, chunksize = 10000, header = None)
    file = pd.concat(tf, ignore_index=True)
# collect information
info = {'column number': [],
        'column name': [],
        'column data type': [],
        'count': [],
        'missing values': [],
        'mean': [],
        'max': [],
        'min': [],
        'standard dev': []
        }
info['column number'] = range(1, length+1)
info['column name'] = list(file.columns)
info['column data type'] = list(file.dtypes.values)
des = pd.DataFrame(file.describe())
m = 0
for i in range(length):
    nrows = len(file.ix[:,i])
    if file.dtypes.values[i] == 'O':
        info['mean'].append(None)
        info['min'].append(None)
        info['max'].append(None)
        info['standard dev'].append(None)
    else:
        info['mean'].append(des[(file.columns[i])].ix['mean'])
        info['min'].append(des[(file.columns[i])].ix['min'])
        info['max'].append(des[(file.columns[i])].ix['max'])
        info['standard dev'].append(des[(file.columns[i])].ix['std'])
    miss = file[pd.isnull(file.ix[:,i])]
    missing = len(miss.index)
    m += missing
    info['missing values'].append(missing)
    info['count'].append(nrows - missing)
info_pd = pd.DataFrame(info)
info_pd.set_index('column number', inplace = True )
info_pd = pd.DataFrame.transpose(info_pd)

# check class label
inb = False
if label == 'T':
    balance = {}
    if class_col > len(file.columns) or class_col < 0:
        exit("Column number out of range!")
    label = file.ix[:, class_col]
    for each in label:
        if each in balance:
            balance[each] += 1
        else: balance[each] = 1
    stats += "\nClass Label\t\tCount\n"
    for i in balance:
        stats += str(i)
        stats += "\t\t\t"
        stats += (str(balance[i]) + "\n")
        for j in balance:
            if balance[i]/balance[j] > 10 or balance[i]/balance[j] < 0.1:
                inb = True

# output
infile = infile.split('/')  # if file name includes path
infile = infile[-1]
stats += "\n##Column information##\n"
stats += str(info_pd)
warning = False
stats += "\n\n##Warnings##\n"
if m > 0:
    stats += "Warning: Missing Values observed!\n"
    warning = True
if "O" in file.dtypes.values:
    stats += "Warning: Non-numeric Values observed!\n"
    warning = True
if (inb == True):
    stats += "Warning: Imbalanced class labels!\n\n"
    warning = True
if warning == False:
    stats+= "None. Your data is good to go!\n\n"


# remove missing values
if rmna == 'T':
    file2 = file.dropna()
    outfile = 'rmna_' + infile
    file2.to_csv(outfile, index=False)  # file output
    stats += "##Missing Values##\n"
    stats += str(nrows - len(file2))
    stats += (" lines with missing values have been removed. New data was saved to: " + outfile + "\n\n")

# verbose vs non-verbose mode
if verbose == "T":
    print stats
else:
    f1 = open("file_stats.txt", "w")
    print >> f1, stats
    f1.close()
    if rmna == 'T':
        file2.to_csv(outfile, index=False)
        print file2.to_csv(index=False)
    f2 = open("log.txt", "w")
    print >> f2, "Data Check is DONE."
    f2.close()
