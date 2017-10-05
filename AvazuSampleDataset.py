import pandas as pd
import numpy as np
import os
import random

data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


print('Original data Class 1 prior:'+str(len(data.click[data.click == 1])/len(data)))

#seed = 13
#np.random.set_state(seed)

Length = len(data)

sampleIX = np.sort(np.random.permutation(round(0.002*Length)))
FirstSample = data.iloc[sampleIX,:]
data.drop(data.index[np.arange(0,round(0.002*Length))])

print('1st sample Class 1 prior:'+str(len(FirstSample.click[FirstSample.click == 1])/len(FirstSample)))


LengthSample = len(FirstSample)
Length = len(data)

sampleIX = random.sample(range(Length),LengthSample)
SecondSample = data.iloc[sampleIX,:]

print('2nd sample Class 1 prior:'+str(len(SecondSample.click[SecondSample.click == 1])/len(SecondSample)))

sampleIX = random.sample(range(len(test)),round(LengthSample*0.1))
TestSample = test.iloc[sampleIX,:]

print(len(FirstSample))
print(len(SecondSample))
print(len(TestSample))

print(os.getcwd())
FirstSample.to_csv(os.getcwd() + '/SmallFirstSample.csv', sep=',', index=False)
SecondSample.to_csv(os.getcwd() + '/SmallSecondSample.csv', sep=',', index=False)
TestSample.to_csv(os.getcwd() + '/SmallTestSample.csv', sep=',', index=False)



