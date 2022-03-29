# pj1

This is the first project: to create a two-layer MLP for classification of MNIST images.

model.py: We define some useful functions and the class of MLP;

train.py: We use grid search to tune the hyper-parameters, including the dimension of the hidden layer (h1), the initial learning rate (v0) and L2-regularization 
coefficient (lam). See train.txt for full results.

test.py: We compute the accuracy of our model on the test dataset, with comparison to other machine learning algorithms.

visualize.py:

model.pickel: The trained MLP.
