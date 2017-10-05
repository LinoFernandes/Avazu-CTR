import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

firstsample = pd.read_csv('SmallFirstSample.csv')
train = pd.read_csv('SmallSecondSample.csv')
test = pd.read_csv('SmallTestSample.csv')

fsClicks = firstsample.click.values
Clicks = train.click.values
train = train.drop('click',axis=1)
firstsample = firstsample.drop('click',axis = 1)

Length = len(train)
fsLength = len(firstsample)
data = pd.concat([firstsample,train,test],ignore_index=True)

timestamp = data.hour
auxHour = [int(str(i)[6:8]) for i in timestamp]
data['Hours'] = auxHour

auxDays = [int(str(i)[4:6]) for i in timestamp]
data['Days'] = auxDays

#Creating Features

# User, App, Site
User = [int(data.device_id[i],16) + int(data.device_model[i],16) + int(data.device_ip[i],16) for i in range(0,len(data))]
App = [int(data.app_id[i],16) + int(data.app_domain[i],16) + int(data.app_category[i],16) for i in range(0,len(data))]
Site = [int(data.site_id[i],16) + int(data.site_domain[i],16) + int(data.site_category[i],16)  for i in range(0,len(data))]
data=data.drop(data.columns[np.arange(4,13)],axis=1)
data=data.drop('hour',axis=1)

data['User'] = User
data['App'] = App
data['Site'] = Site

#One Hot
Unique = [len(set(data[i])) for i in data.columns]
UniqueIX = [i for i in range(0,len(Unique)) if Unique[i] < 100]

data=pd.get_dummies(data,columns=data.columns.values[UniqueIX],drop_first=True)

#History

fsData = data.loc[0:fsLength-1]
train = data.loc[fsLength:(fsLength+Length)-1]
test = data.loc[(fsLength+Length):len(data)]

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

NB = GaussianNB()
RF = RandomForestClassifier(n_estimators=50)
LR = LogisticRegression()

NB.fit(fsData.values, fsClicks)
RF.fit(fsData.values, fsClicks)
LR.fit(fsData.values, fsClicks)

ScoresNB = NB.predict_proba(train.values)
ScoresRF = RF.predict_proba(train.values)
ScoresLR = LR.predict_proba(train.values)

History = []
for i in range(0,len(ScoresNB)):
    History.append(round(np.mean([ScoresNB[i][1],ScoresRF[i][1],ScoresLR[i][1]]),4))

train['History'] = History
test['History'] = 0

UniqueUser = list(set(train.User))

for user in UniqueUser:
    ix = test.User.index[test.User == user]
    if len(ix) > 0:
        test.History[ix] = round(np.mean(train.History.loc[train.User == user]),4)



train['click'] = Clicks

train.to_csv(os.getcwd() + '/SmallTrainPreprocessedV2.csv', sep=',', index=False)
test.to_csv(os.getcwd() + '/SmallTestPreprocessedV2.csv', sep=',', index=False)







