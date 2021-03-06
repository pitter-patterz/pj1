# Introduction

This is the first project: to create a two-layer MLP for classification of MNIST images.

*model.py:*  We define the class of MLP and some useful functions.

*data_process.py:*  Load the MNIST dataset. Transform images to feature vectors and divide them into train/valid/test dataset.

*train.py:*  Use grid search to tune the hyper-parameters, including the dimension of the hidden layer (h1), the initial learning rate (v0) and L2-regularization 
coefficient (lam). Parameters are updated via SGD+Adagrad. See train_record.txt for full results.

*train_byPytorch.py:*  We train a three-layer MLP by Pytorch, with hidden layers of higher dimension. We choose Adam optimizer.

*test.py:*  We compute the accuracy of our model on the test dataset, with comparison to other machine learning algorithms. We also examine some of the misclassified images.

*vis.py:*  We use heat matrix to visualize the model parameters and input/output of the hidden layer.

*model.pickel:*  The trained MLP (using numpy).

*model_Pytorch.pkl:*  The trained MLP (using Pytorch). 

Models can be downloaded from https://pan.baidu.com/s/1fLbs9rh4dFOmK6e921Gslg?pwd=sjwl (password: sjwl).

# Usage

python train.py

python train_byPytorch.py

python test.py

python vis.py

