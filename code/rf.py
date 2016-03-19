# This script runs cross validation + random forest from sklean
import numpy as np
import pandas as pd
import sys
import getopt
import operator
import time
import os
from sklearn import metrics
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier


if len(sys.argv) <= 3:
    usage = "\n##Usage## \npython rf.py [-i input_file] -a col_numbers_iv -c col_numbers_dv [-v T] [-h T] \n"
    usage += "-i: input file. This file should be a .csv file, and the file name includes both the path and the .csv\n"
    usage += "    If -i is not specified, standard input will be used as input file.\n"
    usage += "-a: column numbers of independent variables. A numeric list that corresponds to the columns that will be used \n"
    usage += "    to do prediction. Use '-' between the beginning and end of continuous column numbers and ',' between two\n"
    usage += "    non-continuous column numbers. Note: no spaces are allowed.\n"
    usage += "    eg: 1-3 means column 1, 2 and 3; 1-3,5,7 means column 1,2,3,5,7.\n"
    usage += "-c: column numbers of dependent variables: a numeric value that corresponds to class label location.\n"
    usage += "-n: The number of trees in the forest (default is 50);\n"
    usage += "-f: The number of features to consider at each split (default is sqrt(nfeatures));\n"
    usage += "-d: The maxiumn depth of the tree (default is None: trees are expanded until all leaves are pure)).\n"
    usage += "-v: verbose mode. This option carries two values. 'T' indicates verbose mode, 'F' indicates non-verbose mode.\n"
    usage += "    Please use '-v F' if conducted in a pipeline.\n"
    usage += "-h: header. The default value of this option is 'T', meaning column headers are provided in the file.\n"
    usage += "    If the file contains no column header, use '-h F'.\n"
    exit(usage)

def preProcess(infile, attr, label, header, verbose):
    if header == "T":
        tf = pd.read_csv(infile, iterator = True, chunksize = 10000)
        train = pd.concat(tf, ignore_index=True)
    else:
        tf = pd.read_csv(infile, iterator = True, chunksize = 10000, header = None)
        train = pd.concat(tf, ignore_index=True)

    columns = []
    attr = attr.split(",")
    for each in attr:  # create a list with all the names of the desired columns
        if "-" not in each:
            columns.append(int(each)-1) #(colnames[int(each)-1])
        if "-" in each:
            a,b = each.split("-")
            for num in range(int(a)-1,int(b)):
                columns.append(num)
    if len(columns) == 1:
        train_attr = train.ix[:, columns[0]]
    else:
        train_attr = train.ix[:,columns]
    train_label = train.ix[:, int(label)-1]
    return [train_attr, train_label]

def isinteger(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def rf(train_attr, train_label, n_estimators = 50, max_features = 'auto', max_depth = None):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators = n_estimators, max_features = max_features, max_depth=max_depth)
    nsamples = len(train_attr.index)
    cv = cross_validation.ShuffleSplit(nsamples, n_iter = 2, test_size = 0.2)
    scores = cross_validation.cross_val_score(clf, train_attr,train_label, cv=cv)
    print 'scores: ', scores, len(scores)
    sm = round(np.mean(scores),4)
    sstd = round(np.std(scores),4)
    out = str(sm) + ' +/- ' + str(sstd*2)
    clf.fit(train_attr,train_label)
    return[clf, out]

opts, args = getopt.getopt(sys.argv[1:], "i:a:c:v:h:n:d:f:")

header = "T"
verbose = "T"
attr = -1
label = -1
number = 0
max_depth = None
max_feather = 'auto'  ## default values

log = time.strftime("%c")  # print time stamp
log += "\n#####  Model selection  ##### \n"

for o,a in opts:
    if o == '-i':
        infile = a
    elif o == '-a':
        attr = a
    elif o == '-c':
        if not isinteger(a):
            exit("The class label has to be followed by a whole number.")
        label = a
    elif o == "-v":
        verbose = a.upper()
        if verbose not in ["T", "F"]:
            exit ("The verbose option is not properly provided! Please type '-v F' or '-v T'")
    elif o == '-h':
        header = a.upper()
        if header not in ['T', 'F']:
            exit("The header option is not properly provided! Please type '-h F' or '-h T'.")
    elif o == '-n':
        if not isinteger(a):
            exit("The number of trees must be an integer!")
        if a <= 0:
            exit("The number of trees must be positive!")
        number = int(a)
    elif o == '-d':
        if isinteger(a):
            max_depth = int(a)
        else: max_depth = a
    elif o == '-f':
        if isinteger(a):
            max_feather = int(a)
        else: max_feather = a


log = time.strftime("%c")
log += "\nTraining data: " + infile
train_attr, train_label = preProcess(infile, attr, label, header, verbose)

if number == 0:
    n_estimator = 50
else: n_estimator = number
clf, acc = rf(train_attr, train_label, n_estimator, max_depth=max_depth, max_features= max_feather)
log += "\nPredictive Model: Random Forest\n"
log += "Parameters: "
log += "Iterations: {0}; max depth: {1}; max_features: {2}".format(n_estimator, max_depth, max_feather)
log += "\nModel accuracy with cross validation: "
log += acc

if verbose == "T":
    print log
else:
    f1 = open("training_log.txt", "a")
    print >> f1, log
    print >> f1, "Model Selection is DONE.\n\n"
    f1.close()
