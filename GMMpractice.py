#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 13:16:48 2017

@author: deaxman
"""

#%%
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from numpy.linalg import inv
from numpy.linalg import det
#%%
S=np.array([1,1,1,2,2,3,3,3,2,4,4,4])-1
#X=np.array([[0,0],[-1,0],[0,1],[3,3],[3,4],[7,7],[7,8],[8,7],[2,3],[12,10],[10,10],[11,10]])
X=np.random.rand(50,2)*10
plt.scatter(X[:,0],X[:,1])
k=4
centers=X[np.random.choice(X.shape[0],size=(4),replace=False),:]
sigmas=np.array([np.eye(2) for i in range(k)])
sigmaInvs=np.array([inv(sigmas[i,:,:]) for i in range(k)])
Z=np.zeros((200,200))
def createCovMat(X,w):
    print(np.dot(X.T*w.T,X)/(np.sum(w[0])))
    return np.dot(X.T*w.T,X)/(np.sum(w[0]))

def gaussian2d(X,mu,sigma):
    return np.exp(-0.5*np.dot(np.dot((X-mu),inv(sigma)),(X-mu).T))/(2*np.pi*det(sigma))
    
for i in np.arange(5):
    W=np.exp(-0.5*np.sum((X[:,:,np.newaxis]-centers[:,:,np.newaxis].T).T*np.sum((sigmaInvs[:,:,:,np.newaxis]
        *(X[:,:,np.newaxis]-centers[:,:,np.newaxis].T).T[:,:,:,np.newaxis].transpose((0,1,3,2))),axis=1),axis=1).T)/((2*np.pi*np.array([np.sqrt(det(sigmas[i,:,:])) for i in range(k)]))[:,np.newaxis].T)
    
    #/((2*np.pi*np.array([np.sqrt(det(sigmas[i,:,:])) for i in range(k)]))[:,np.newaxis].T)
    l=np.argmin(W,axis=1)
    fig=plt.figure()
    ax=plt.axes()
    ax.scatter(X[:,0],X[:,1])
    a, b = np.meshgrid(np.arange(-5,15,0.1), np.arange(-5,15,0.1))
    for blob in range(centers.shape[0]):
        for j in range(200):
            for i in range(200):
                Z[j,i]=gaussian2d(np.concatenate((a[:,:,np.newaxis],b[:,:,np.newaxis]),axis=2)[i,j,:],centers[blob,:],sigmas[blob,:])
        ax.contour(a,b,Z)
    ax.scatter(centers[:,0],centers[:,1],marker='x')
    centers=np.array([np.sum(X*(W[:,i])[:,np.newaxis],axis=0)/np.sum(W[:,i],axis=0) for i in range(k)])
    sigmas=np.array([createCovMat(X,(W[:,i])[:,np.newaxis]) for i in range(k)])
    sigmaInvs=np.array([inv(sigmas[i,:,:]) for i in range(k)])
    
    
    
