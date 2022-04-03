# pj1

This is the first project: to create a two-layer MLP for classification of MNIST images.

model.py: We define the class of MLP and some useful functions.

data_process.py: Load the MNIST dataset. Transform images to feature vectors and divide them into train/vaid/test set.

train.py: Use grid search to tune the hyper-parameters, including the dimension of the hidden layer (h1), the initial learning rate (v0) and L2-regularization 
coefficient (lam). Parameters are updated via SGD+Adagrad. See train_record.txt for full results.

train_byPytorch: We train a three-layer MLP by Pytorch, with higher dimension of hidden layers. We choose Adam optimizer.

test.py: We compute the accuracy of our model on the test dataset, with comparison to other machine learning algorithms. We also examine some of the misclassified images.

visualize.py:

model.pickel: The trained MLP (using numpy).

Pytorch_model.pkl: The trained MLP (using Pytorch). 

command: 
> python train.py
> python train_byPytorch.py
> python test.py

