import pickle,random,time,matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from model import *
from data_process import *

xtr,ytr,xv,yv,xte,yte = load_data()
n = xte.shape[0]

with open('model.pickle','rb') as file:
    mlp = pickle.load(file)
    
yhat = mlp.forward(xte)
accuracy = cal_acc(yhat,yte)
print('\n\nAccuracy of MLP',accuracy)

label = np.argsort(yhat)[:,-1].reshape(n,1)
err_index = [i for i in range(n) if label[i,0]!=yte[i,0]]

print('\n\nShow some of the misclassified images---\n\n')
time.sleep(2)

for i in random.sample(err_index,5):
    true_label = yte[i,0]
    mis_label = label[i,0]
    X = xte[i,:].reshape(28,28)
    plt.figure(dpi=300,figsize=(6,3))
    plt.imshow(X)
    plt.xlabel('True & Predicted label = '+str(true_label)+' & '+str(mis_label),fontsize=10)
    plt.show()

M = np.zeros(shape=(10,10))
for i in range(10):
    for j in range(10):
        indexs = [k for k in range(n) if yte[k,0]==i]
        indexs = [k for k in indexs if label[k,0]==j]
        M[i,j] = len(indexs)
M /= np.sum(M,axis=1,keepdims=True)

print('\n\nShow the confusion matrix---\n\n')
print(M)

print('\n\nFit a logistic regression model---\n\n')
lr = LogisticRegression() 
lr.fit(xtr,ytr)
print('\n\nAccuracy of logistic regression',lr.score(xte,yte))

print('\n\nFit a random forest model (n_estimators=100,max_depth=5)---\n\n')
rf = RandomForestClassifier(n_estimators=100,max_depth=5)
rf.fit(xtr,ytr)
print('\n\nAccuracy of Random Forest',rf.score(xte,yte))

from train_byPytorch import *
mlp_Pytorch = torch.load('model_Pytorch.pkl')

xte,yte = torch.tensor(xte).float(),torch.tensor(yte).long().squeeze()  
accuracy = torch_acc(xte,yte,mlp_Pytorch)
print('\n\nAccuracy of MLP (trained by Pytorch)',accuracy)
