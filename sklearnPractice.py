#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 13:31:10 2017

@author: deaxman
"""

#%%
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
#%%
iris = sns.load_dataset('iris')
iris.head()
#%%
sns.pairplot(iris,hue='species',size=1.5)
#%%
X_iris = iris.drop('species',axis=1)
y_iris=iris['species']
#%%
rng = np.random.RandomState(42)
x = 10 * rng.multivariate_normal(np.zeros((2,)),np.array([[1,.99],[.99,1]]),50)
sns.kdeplot(pd.DataFrame(x),shade=True)
y = 2 * x[:,0]+3*x[:,1] - 1 + rng.randn(50)
#plt.scatter(x, y);
#x=x[:,np.newaxis]
model=LinearRegression()
randSamp=rng.choice(x.shape[0],size=50)
model.fit(x[randSamp,:],y[randSamp])
print(model.intercept_)
#%%
m=np.zeros((100,2))
b=np.zeros((100,1))
model=LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
for i in range(100):
    randSamp=rng.choice(x.shape[0],size=50)
    model.fit(x[randSamp,:],y[randSamp])
    m[i,:]=model.coef_
    b[i]=model.intercept_
sns.kdeplot(pd.DataFrame(m),shade=True)
#%%
from sklearn.cross_validation import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris,
                                                random_state=1)
from sklearn.decomposition import PCA  # 1. Choose the model class
model = PCA(n_components=2)            # 2. Instantiate the model with hyperparameters
model.fit(X_iris)                      # 3. Fit to data. Notice y is not specified!
X_2D = model.transform(X_iris)
#%%
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree),
                         LinearRegression(**kwargs))
    
#%%
def make_data(N, err=1.0, rseed=1):
    # randomly sample the data
    rng = np.random.RandomState(rseed)
    X = rng.rand(N, 1) ** 2
    y = 10 - 1. / (X.ravel() + 0.1)
    if err > 0:
        y += err * rng.randn(N)
    return X, y

X, y = make_data(500)
plt.scatter(X,y)
#%%

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score
allScores=np.zeros((100,4))
allModels=[]
for i,trainPercent in enumerate(np.linspace(10,95,100)):
    for j,degree in enumerate(np.arange(2,6)):
        tmpScores=[]
        for tmp in range(5):
            Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,train_size=(trainPercent/100))
            modelFIT=PolynomialRegression(degree).fit(Xtrain, ytrain)
            #allModels.append(modelFIT)
            tmpScores = np.append(tmpScores, r2_score(ytest, modelFIT.predict(Xtest)))
        allScores[i,j] = np.mean(tmpScores)
        print(np.var(tmpScores))
        
sns.heatmap(allScores)
#plt.plot(PolynomialRegression(degree).fit(Xtrain, ytrain).predict(np.linspace(np.min(X),np.max(X),1000)[:,np.newaxis]))
#%%

