# Sample script naive benchmark that yields 0.609 public LB score WITHOUT any image information

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


trainCSVPath = '../input/train.csv'
testCSVPAth = '../input/sample_submission.csv'

train = pd.read_csv(trainCSVPath)
submit = pd.read_csv(testCSVPAth)
Y = train
# convert numeric labels to binary matrix
def to_bool(s):
    return(pd.Series([1 if str(i) in str(s).split(' ') else 0 for i in range(9)]))
Y = train['labels'].apply(to_bool)
Y['bid'] = train['business_id']
print Y
print train
exit()
# get means proportion of each class
py = Y.mean()
plt.bar(Y.columns,py,color='steelblue',edgecolor='white')

# predict classes that are > 0.5, 2,3,5,6,8
submit['labels'] = '2 3 5 6 8'
submit.to_csv('../input/naive.csv',index=False)

trainCSVPath = '../input/train.csv'
testCSVPAth = '../input/sample_submission.csv'
naivePath = '../input/naive.csv'

train = pd.read_csv(trainCSVPath)
submit = pd.read_csv(testCSVPAth)
naive = pd.read_csv(naivePath)
print naive[0:1]

print util.getAccuracy(submit, naive)