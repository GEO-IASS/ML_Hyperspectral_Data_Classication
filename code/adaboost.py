__author__ = 'penghao'
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

infile = np.load('train.npy')
testfile = np.load('test.npy')
x=infile[:,:200]
y= infile[:,200]
test_x=testfile[:,:200]
test_y=testfile[:,200]


clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4),n_estimators=500, learning_rate=1)
#scores = cross_val_score(clf, x, y)
clf.fit(x,y)
#print(scores)

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