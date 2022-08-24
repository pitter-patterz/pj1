The following files are provided.


*model.py:*  We define the class of MLP and some useful functions.

*data_process.py:*  Load the MNIST dataset. Transform images to feature vectors and divide them into train/valid/test dataset.

*train.py:*  Use grid search to tune the hyper-parameters, including the dimension of the hidden layer (h1), the initial learning rate (v0) and L2-regularization 
coefficient (lam). Parameters are updated via SGD+Adagrad. See train_record.txt for full results.

