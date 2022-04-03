"""
Load the MNIST dataset.
Transform each image to a 28*28=784 1d array, as the input of MLP.
Samples are divided into train/valid/test dataset, under the batch-mode.

"""

import numpy as np
from sklearn.model_selection import train_test_split,KFold

def load_data():

    A = np.load('mnist.npz')
    x1,x2 = A['x_train'],A['x_test']
    y1,y2 = A['y_train'],A['y_test']
    y1,y2 = np.asarray([y1]).T,np.asarray([y2]).T
    
    m1 = [x1[i].flatten() for i in range(x1.shape[0])]
    m2 = [x2[i].flatten() for i in range(x2.shape[0])]
    m1,m2 = np.asarray(m1),np.asarray(m2)
    
    xtr,ytr = m1,y1
    xte,yte = m2,y2
    
    np.random.seed(123)
    xtr,xv,ytr,yv = train_test_split(xtr,ytr,test_size=0.2,shuffle=True)
    return xtr,ytr,xv,yv,xte,yte


def batch(x,y,size):
    
    np.random.seed(123)
    n = x.shape[0]
    split = int(n/size)
    kf = KFold(n_splits=split,shuffle=True)
    
    seq = []
    for i,j in kf.split(x):
        seq.append(j)
        
    B = [(x[s,:],y[s,:]) for s in seq]
    return B