"""
Train a MLP within 200 epochs. 
Use grid search to determine the tuning parameters.
"""

import numpy as np
from model import *
from data_process import *

xtr,ytr,xv,yv,xte,yte = load_data()
B = batch(xtr,ytr,size=20)

dimx = xtr.shape[1]
dimy = np.max(ytr)+1

def train(h1,v,lam):
    
    np.random.seed(123)
    m = mlp(dimx=dimx,dimy=dimy,h1=h1,v=v,lam=lam)
    losses,accs = [],[]
    
    for epoch in range(200):
          
        yhat = m.forward(xte)
        accuracy = cal_acc(yhat,yte)
        loss = m.cal_loss(xte,yte)
        losses.append(loss)
        accs.append(accuracy)
        print('epoch:',epoch,'loss:',loss,'accuracy:',accuracy)
        # print('epoch:',epoch,'loss:',loss,'accuracy:',accuracy,file=f)
        
        for x0,y0 in B:
            m.sgd_step(x0,y0)
             
    return losses,accs,m


for v in [7e-3,8e-3,10e-3]:
    for lam in [1e-4,5e-4,10e-4]:
        for h1 in [32,64,128]:            
            tune = (h1,v,lam)
            
            # fname = str(h1)+'_'+str(int(v*1e3))+'_'+str(int(lam*1e4))+'.txt'
            # fname = 'new.txt'
            # f = open(fname,'w')
            # print('\nTuning Parameters (h1,v,lam):',tune,file=f)
            
            print('\nTuning Parameters:',tune)
            
            losses,accs,m = train(h1=h1,v=v,lam=lam)
            
            print('Best accuracy on validation data',max(accs))
            # print('Best accuracy on validation data',max(accs),file=f)
            
            # f.close()
            
# m.save('.pickle')
