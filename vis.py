
import numpy as np,pickle,time
import matplotlib.pyplot as plt,matplotlib.cm as cm
from model import *
from data_process import *

with open('model.pickle','rb') as file:
    m = pickle.load(file)

def addcol(x,t):
    x0 = x.copy()
    for i in range(t):
        x0 = np.hstack((x0,x))
    return x0


def tosquare(x,t):
    x0 = x.copy()
    S = []
    i = 0
    while True:
        if i+t >= x0.shape[0]:
            break
        xx = x0[i:i+t,:]
        S.append(np.sum(xx,axis=0))
        i += t
    S = np.asarray(S)
    return S


print('\nWe implement heatmap to visualize the parameters of MLP---\n')
time.sleep(1)

print('\n Initial network---\n')
time.sleep(1)

m0 = mlp()
plt.figure(dpi=300,figsize=(8,3))
plt.subplot(1,4,1)
plt.imshow(m0.w1,cmap=cm.hot)
plt.colorbar()
plt.xlabel('weight_1')

plt.subplot(1,4,2)
plt.imshow(m0.w2,cmap=cm.hot)
plt.colorbar()
plt.xlabel('weight_2')

plt.subplot(1,4,3)
plt.imshow(addcol(m0.b1.T,15),cmap=cm.hot)
plt.colorbar()
plt.xlabel('bias_1')

plt.subplot(1,4,4)
plt.imshow(addcol(m0.b2.T,1),cmap=cm.hot)
plt.colorbar()
plt.xlabel('bias_2')
plt.show()

print('\n After training---\n')
time.sleep(1)

plt.figure(dpi=300,figsize=(8,3))
plt.subplot(1,4,1)
plt.imshow(m.w1,cmap=cm.hot)
plt.colorbar()
plt.xlabel('weight_1')

plt.subplot(1,4,2)
plt.imshow(m.w2,cmap=cm.hot)
plt.colorbar()
plt.xlabel('weight_2')

plt.subplot(1,4,3)
plt.imshow(addcol(m.b1.T,15),cmap=cm.hot)
plt.colorbar()
plt.xlabel('bias_1')

plt.subplot(1,4,4)
plt.imshow(addcol(m.b2.T,1),cmap=cm.hot)
plt.colorbar()
plt.xlabel('bias_2')
plt.show()


xtr,ytr,xv,yv,xte,yte = load_data()
n = xte.shape[0]

print('\nInput and output of the hidden layer---\n')
time.sleep(1)

for i in range(4):
    idx = np.random.randint(0,n)
    x = xte[idx,:].reshape(1,28**2)
    img = x.reshape(28,28)
    
    z1 = x.dot(m.w1)+m.b1
    a1 = sigmoid(z1)
    z2 = a1.dot(m.w2)+m.b2
    e = np.exp(z2)
    sume = np.sum(e,axis=1)
    prob = e/sume
        
    yhat = m.forward(x)
    sort_num = np.argsort(yhat)[0]
    
    title_str = ''
    for i in range(3):
        num = sort_num[-(i+1)]
        title_str += 'P(N='+str(num)+')='+str(round(yhat[0,num],4))
        if i != 2:
            title_str += ', '
    
    
    plt.figure(dpi=300,figsize=(6,8))
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.xlabel('Original Image')
    plt.subplot(1,3,2)
    plt.imshow(z1.reshape(16,8))
    plt.xlabel('Input of layer1')
    plt.title(title_str,y=1.05)
    plt.subplot(1,3,3)
    plt.imshow(a1.reshape(16,8))
    plt.xlabel('Output of layer1')
    
    plt.show()


