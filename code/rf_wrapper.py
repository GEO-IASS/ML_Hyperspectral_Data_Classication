# this script needs to be executed from dropbox/cs578project/analysis

import subprocess
iter = [20,50,100,150,200,500]
nfeature = range(3,11)
total = len(iter) * len(nfeature)
for i in iter:
    for j in nfeature:
        print 'round ', str(i*len(nfeature)+j), 'of ', str(total)
        command = 'python ../code/rf.py -i ../data/train.csv -a 1-200 -c 201 -n {0} -f {1} -v F'.format(i, j)
        output = subprocess.Popen(command.split(),shell=False,stdout=subprocess.PIPE)

print 'Done'
