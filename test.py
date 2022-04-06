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
    plt.figure(dpi=300,figsize=(3,3))
    plt.imshow(X)
    plt.title('True & Predicted label = '+str(true_label)+' & '+str(mis_label),fontsize=10)
    plt.show()
    
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
