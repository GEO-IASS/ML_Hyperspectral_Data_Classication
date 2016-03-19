################################################################################
# Mind Fabric Pipeline
# Step 1.2: Feature selection and dimensionality reduction -- pca
# Copyright (c) SparkCognition Inc. 2014
# Author: Dan Wang
# July 28, 2015
# Version 0.1
# This is the command line compatible version
# Input: original csv file with no missing values
# File output: 1. scree plot (filename_scree.png)
#              2. variance percentage explained (filename_pc_variance.csv)
#              3. principle components (filename_pc.csv)
################################################################################

#! /usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import subprocess
import sys
import getopt
import csv

if len(sys.argv) < 3:
    usage = "\n##Usage##\npython pca.py -i input_file [-h T] [-p percent/components] [-c class_label] [-m method#]\n"
    usage += "-i: input_file. This file should be a .csv file, and the file name includes both the path and the .csv\n"
    usage += "-h: header. The default value of the -h option is T, meaning column headers are provided in the file.\n"
    usage += "    If the file contains no column header, use '-h F'.\n"
    usage += "-c: class_label. A numeric value that corresponds to column number of the class label. If -c is not given,\n"
    usage += "    the last column will be used as class label.\n"
    usage += "-m: method#. A numeric value that corresponds to the type of pca. 1: covariance; 2: correlation. In other\n"
    usage += "    words, '-m 1' indicates the mean centering of the original data, while '-m 2' indicates both the mean\n"
    usage += "    centering and scaling of the original data. The default choice is 1.\n"
    usage += "-p: percentage/components. If a real number between 0 and 1 is given, this number will be treated as the\n"
    usage += "    percentage of variance that the new principle components explain.\n"
    usage += "    If an integer >= 1 and <= (ncol of the input file) is provided, this number will be treated as the number\n"
    usage += "    of components that will be kept in the output file.\n"
    usage += "    If the -p option is not provided, -p all will be used, meaning all components will be kept and the output \n "
    usage += "    csv file have the same number of columns and the same number of rows as the input csv file.\n"
    usage += "-v: verbose mode. T indicates verbose mode, F indicates non-verbose mode.\n"
    usage += "    Please use '-v F' if conducted in a pipeline.\n"
    exit(usage)

def file2DF(infile, header):
    if header == "T":
        tf = pd.read_csv(infile, iterator = True, chunksize = 10000)
        df = pd.concat(tf, ignore_index=True)
    else:
        tf = pd.read_csv(infile, iterator = True, chunksize = 10000, header = None)
        df = pd.concat(tf, ignore_index=True)
    return df

def parseFilename(infile):
    if '/' in infile:
        infile = infile.split('/')
        infile = infile[-1]
    if '.csv' in infile:
        filenames = infile.split('.')
        del filenames[-1]
        filename = ''.join(filenames)
    else:
        exit('Please provide a valid input file name.')
    return filename

def slice(df, label):
    dfdv = df.ix[:, label]
    dfiv = df.drop(df.columns[label], 1, inplace = False)
    #dfdv = dfdv.convert_objects(convert_numeric = True)
    dfiv = dfiv.convert_objects(convert_numeric = True)
    return [dfiv, dfdv]

def scale(df):
    from sklearn import preprocessing
    df = df.convert_objects(convert_numeric = True)
    df_scaled = preprocessing.scale(df)
    return df_scaled

def pca_variance(df):  # inputs are original data frame
    df_pca = PCA()
    df_pca.fit(df)
    ratio = df_pca.explained_variance_ratio_
    components = [('component'+str(x)) for x in range(1, (df.shape[1]+1))]
    df2 = pd.Series(ratio, index = components)
    return df2

def effectComp(df2, comp):  # input is variance table and comp parsed from command line options
    if comp < 1:
        var = 0
        require = '{} % of the variance are required to be explained by the principal components.'.format(comp*100)
        for i in range(df2.shape[0]):
            var += df2.ix[i,:]
            if var >= comp:
                comp = i + 1
                break
    elif comp == 'all':
        require = 'All principal components are kept.'
        comp = df2.shape[0]
    else:
        comp = int(comp)
        require = '{} components are required to keep.'.format(comp)
    return [comp, require]

def writePC(df, comp): # input is effective Components calculated from effectComp, and the original df
    if comp > df.shape[1]:
        exit('The required components number is larger than the number of columns in the original csv file.')
    df_pca = PCA(comp)
    out = df_pca.fit_transform(df)
    return out

def screePlot(df2):  # input is the variance df
    fig = plt.figure()
    plt.plot(range(1, df2.shape[0]+1), df2.values, 'ro-', linewidth = 2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Components')
    plt.ylabel('Percentage Variance')
    return fig

def isfloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
####################################################################################################################
opts, args = getopt.getopt(sys.argv[1:], "p:h:i:c:m:v:")
header = "T"
infile = ""
comp = 'all'
label = 'last'
verbose = 'T'
method = 1       # defaults
for o,a in opts:
    if o == '-i':
        infile = a
    elif o == '-h':
        header = a.upper()
        if header not in ['T', 'F']:
            exit("The header option is not properly provided! Please type '-h F' or '-h T'.")
    elif o == '-p':
        if isfloat(a):
            if float(a) < 1 and float(a) > 0:
                comp = float(a)
            elif float(a) >= 1 and float(a) % 1 ==0:
                comp = int(a)
            else:
                exit('-p option accepts either a real number between 0 and 1 or an interger larger than 1 and smaller than the number of columns in the original csv file.')
        else:
            exit('-p option accepts only numeric inputs.')
    elif o == '-c':
        if not isfloat(a):
            exit("Column number of the class label needs to be a positive integer.")
        label = int(a)
    elif o == '-m':
        if a not in ['1', '2']:
            exit("Please choose between 1 (covariance PCA) or 2 (correlation PCA) for -m option.")
        method = int(a)
    elif o == '-v':
        verbose = a.upper()
        if verbose not in ["T", "F"]:
            exit ("The verbose option is not properly provided! Please type '-v F' or '-v T'")

df = file2DF(infile, header)
filename = parseFilename(infile)
last_col = df.shape[1]

if label == 'last':  # default: last column is the class label
    label = last_col
elif label <= 0 and label > last_col:
    exit('The column index of the class label is out of the range.')
label = label - 1  # 0 based
df, dflabel = slice(df, label) # separate class label and the rest
#print df.shape


if method == 2:   # correlation PCA
    df = scale(df)
    
df2 = pca_variance(df)  # percentage variance table
components = df2.index.values
scree = screePlot(df2)   # draw the scree plot
comp, require = effectComp(df2, comp)   # effect principal components
components = list(components[0:comp])
components.append('class_label')  # column title of the output csv file
pc_array = writePC(df, comp)    # output the pcs as np.ndarray
pc_array = list(pc_array)
for i in range(len(pc_array)):
    pc_array[i] = list(pc_array[i])
    pc_array[i].append(dflabel[i])
# pc_df = pd.DataFrame(index=range(pc_array.shape[0]), columns=components)
# for i in range(pc_array.shape[0]):   # convert np.ndarray to pandas df
#     pc_df.ix[i,:] = pc_array[i]
f1 = filename + '_scree.png'
f2 = filename + '_pc_variance.csv'
f3 = filename + '_pc.csv'


# write log
log = 'User requirement: ' + require
log +=  'Results: {0} components kept.'.format(comp)
log +=  'Class label is in column {0}'.format(label+1)
log += 'The scree plot was saved as {0}; \nThe variance table was saved as {1};\nThe new principal components was saved as {2}'.format(f1, f2, f3)

# write output file
plt.savefig(f1)
#plt.show(scree)
df2.to_csv(f2)
if verbose == 'T':
    fout = open(f3, 'wb')
    a = csv.writer(fout)#, dialect='excel')
    print log
else:
    f4 = open("log.txt", "a")
    print >> f4, log
    print >> f4, "Forecast is Done."
    f4.close()
    a = csv.writer(sys.stdout)
a.writerow(components)
a.writerows(pc_array)






