import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('SmallSecondSample.csv')
#
timestamp = data.hour
auxHour = [int(str(i)[6:8]) for i in timestamp]
data['Hour'] = auxHour

auxDays = [int(str(i)[4:6]) for i in timestamp]
data['Days'] = auxDays

features = data.columns[data.columns != 'click']

for feature in features:
    dataCounts = data.groupby(feature).click.count()
    datarate = (dataCounts.values/sum(dataCounts))*100
    plt.plot(datarate,color='r')
    #plt.xticks([])
    plt.xlabel(feature)
    plt.ylabel('Distribution(%)')
    plt.savefig(feature+'-datarate.png')
    plt.close()
