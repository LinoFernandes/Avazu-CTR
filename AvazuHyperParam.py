import pandas as pd
import numpy as np

data = pd.read_csv('SmallTrainPreprocessed.csv')
#test = pd.read_csv('SmallTestPreprocessed.csv')

seeds = [496, 5992, 4394, 1793, 4499]

X = data.ix[:, data.columns != 'click'].values
Y = data.click.values

#for seed in seeds:
from sklearn.model_selection import train_test_split, RandomizedSearchCV
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.25, random_state=496)

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=20)

param_dist = {"n_estimators": np.arange(25,500,20), "criterion": ["gini", "entropy"], "max_features": np.arange(10,200,20)}

scoring = {"Precision": 'precision', "Recall": 'recall', "ROC": 'roc_auc'}

# run randomized search
n_iter_search = 15
random_search = RandomizedSearchCV(RF, param_distributions=param_dist, n_iter=n_iter_search,scoring='roc_auc')
RF_Hyper = random_search.fit(X_Train, Y_Train)

import _pickle as cPickle
# save the classifier
with open('RF_Hyper.pkl', 'wb') as fid:
    cPickle.dump(RF, fid)



