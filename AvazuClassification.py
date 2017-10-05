import pandas as pd
import numpy as np
from scipy import interp
import _pickle as cPickle
import matplotlib.pyplot as plt

data = pd.read_csv('SmallTrainPreprocessed.csv')
#test = pd.read_csv('SmallTestPreprocessed.csv')

seeds = [496, 5992, 4394, 1793, 4499]


X = data.ix[:, data.columns != 'click'].values
Y = data.click.values

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, classification_report

i = 0
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

for seed in seeds:
    print('ROUND:' + str(i))
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.25, random_state=seed)
    #LR = LogisticRegression(penalty= 'l1', C= 10, fit_intercept= True)
    LR = LogisticRegression()
    probas_ = LR.fit(X_Train, Y_Train).predict_proba(X_Test)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(Y_Test, probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    i += 1
    try:
        print(classification_report(Y_Test, LR.predict(X_Test)))
    except:
        continue

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f)' % mean_auc,
         lw=2, alpha=.8)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('LR-ROC')
plt.legend(loc="lower right")
plt.savefig('LR_noParam.png')
plt.show()
