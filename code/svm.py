#!/usr/bin/env python 
#This script run svm from sklearn
#by Jun Chen
import numpy as np
from sklearn import svm
from sklearn import cross_validation
#loading data
infile=np.load("./data/train.npy")
testfile=np.load("./data/test.npy")
x=infile[:,:200]
y=infile[:,200]
test_x=testfile[:,:200]
test_y=testfile[:,200]

#score_max=0.0
#c_f=0;gamma_f=0;
#for c in np.arange(1,10,0.5):
#    for g in np.arange(0,10,0.5):
#        clf=svm.SVC(C=c,gamma=g, kernel='rbf', max_iter=-1,shrinking=True)
#        cv = cross_validation.ShuffleSplit(len(x),test_size =0.2,random_state=0)
#        scores = cross_validation.cross_val_score(clf,x,y=y,cv=cv)
#        print scores.mean()
#        if scores.mean()>score_max:
#            score_max=scores.mean()
#            c_f=c
#            gamma_f=g

c_f=6.5
gamma_f=0.5

clf=svm.SVC(C=c_f,gamma=gamma_f, kernel='rbf', max_iter=-1,shrinking=True)
clf.fit(x,y)

def error(x,y,clf):
#    e=0. #error counter
    e=np.zeros(12) #error counter
    for i in range(len(x)):
        a=clf.predict(x[i])
        if a!=y[i]:
            e[int(y[i]-1)]+=1.
    return e

train_e=error(x,y,clf)
test_e=error(test_x,test_y,clf)

# print the error for each label
print(train_e/100)
print(test_e/100)

#print error for total
print(sum(train_e)/1200)
print (sum(test_e)/1200)
