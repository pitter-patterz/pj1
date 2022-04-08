
"""
sigmoid(),relu(),tanh(): three activation functions. Here we use relu().
cal_acc(): compute the accuracy of classification.

"""

import numpy as np,pickle

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0.0,x)

def tanh(x):
    return np.tanh(x)

def cal_acc(yhat,y):
    n = y.shape[0]
    label = np.argsort(yhat)[:,-1]
    label = label.reshape(n,1)
    accuracy = np.sum(label==y)/n
    return accuracy

class mlp():
    
    def __init__(self,dimx=784,dimy=10,h1=128,v=10e-3,lam=1e-4,p=0.95):

        """
        h1 = dimension of the hidden layer
        dimx, dimy = number of features and labels
        v = (initial) learning rate
        lam = coefficient of L2-regularization 
        cdw1,cdw2,cdb1,cdb2 = cumulative gradient
        p = probability of keeping a neuron = 1-drop_out_rate
        
        """

        self.dimx,self.dimy = dimx,dimy 
        self.h1 = h1 
      
        self.w1 = np.random.uniform(-.1,.1,(dimx,h1))+1e-4
        self.b1 = np.random.uniform(-.1,.1,(1,h1))+1e-4
        self.w2 = np.random.uniform(-.1,.1,(h1,dimy))+1e-4
        self.b2 = np.random.uniform(-.1,.1,(1,dimy))+1e-4
        
        self.v,self.lam = v,lam
        
        self.cdw1 = np.zeros((dimx,h1))
        self.cdb1 = np.zeros((1,h1))
        self.cdw2 = np.zeros((h1,dimy))
        self.cdb2 = np.zeros((1,dimy))
        
        self.p = p
    
    def forward(self,x):
        
        n = x.shape[0]
        z1 = x.dot(self.w1)+self.b1
        a1 = relu(z1)
        
        a1 *= self.p
        
        z2 = a1.dot(self.w2)+self.b2
        e = np.exp(z2)
        sume = np.sum(e,axis=1).reshape(n,1)  

        return e/sume
    
    def cal_loss(self,x,y):
        
        n = x.shape[0]
        yhat = self.forward(x)
        ls = 0
        for i in range(n):
            ls += -np.log(yhat[i,y[i,0]])
        ls += 0.5*self.lam* (np.sum(self.w1**2) + np.sum(self.w2**2))
        return ls/n
    
    def sgd_step(self,x,y):
        
        n = x.shape[0]
        i = np.random.randint(n)
        x0 = x[i,:].reshape(1,self.dimx)
        y0 = y[i,0]
        
        z1 = x0.dot(self.w1)+self.b1
        a1 = relu(z1)
                
        dropif = np.random.rand(*a1.shape) < self.p
        a1 *= dropif
        
        z2 = a1.dot(self.w2)+self.b2  
        e = np.exp(z2)
        sume = np.sum(e)
        e /= sume
        
        delta = e.copy()
        delta[0,y0] -= 1
        
        dw2 = a1.T.dot(delta)
        db2 = np.sum(delta,axis=0,keepdims=True)
        
        relu_grad = (a1>0).reshape(a1.shape)
        
        delta = (delta.dot(self.w2.T)) * (relu_grad)
        
        dw1 = x0.T.dot(delta)
        db1 = np.sum(delta,axis=0,keepdims=True)
        
        dw1 += self.lam*self.w1/n
        dw2 += self.lam*self.w2/n
        
        self.cdw1 += dw1**2
        self.cdw2 += dw2**2
        self.cdb1 += db1**2
        self.cdb2 += db2**2
        

        self.w1 -= self.v*dw1 / (1e-7+self.cdw1**0.5)
        self.b1 -= self.v*db1 / (1e-7+self.cdb1**0.5)
        self.w2 -= self.v*dw2 / (1e-7+self.cdw2**0.5)
        self.b2 -= self.v*db2 / (1e-7+self.cdb2**0.5)
    
 
    def save(self,fname):
        f = open(fname,'wb')
        pickle.dump(self,f)
        f.close()
    
