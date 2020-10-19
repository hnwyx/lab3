from nn.layers import *
from nn.model import Model
from nn.optimizers import RMSprop, Adam
import matplotlib.pyplot as plt
import numpy as np

from models.MNISTNet import MNISTNet
from nn.loss import SoftmaxCrossEntropy, L2
from nn.optimizers import Adam
from data.datasets import MNIST
np.random.seed(5242)

def MyModel():
    conv1_params = {
        'kernel_h': 3,
        'kernel_w': 3,
        'pad': 0,
        'stride': 1,
        'in_channel': 1,
        'out_channel': 10
    }
    conv2_params = {
        'kernel_h': 3,
        'kernel_w': 3,
        'pad': 0,
        'stride': 1,
        'in_channel': 10,
        'out_channel': 24
    }
    pool1_params = {
        'pool_type': 'max',
        'pool_height': 2,
        'pool_width': 2,
        'stride': 2,
        'pad': 0
    }
    pool2_params = {
        'pool_type': 'max',
        'pool_height': 3,
        'pool_width': 3,
        'stride': 2,
        'pad': 0
    }
    model = Model()
    model.add(Conv2D(conv1_params, name='conv1',
                          initializer=Gaussian(std=0.001)))
    model.add(ReLU(name='relu1'))
    model.add(Pool2D(pool1_params, name='pooling1'))
    model.add(Conv2D(conv2_params, name='conv2',
                          initializer=Gaussian(std=0.001)))
    model.add(ReLU(name='relu2'))
    model.add(Pool2D(pool2_params, name='pooling2'))
    # model.add(Dropout(ratio=0.25, name='dropout1'))
    model.add(Flatten(name='flatten'))
    model.add(Linear(600, 256, name='fclayer1',
                      initializer=Gaussian(std=0.01)))
    model.add(ReLU(name='relu3'))
    # model.add(Dropout(ratio=0.5))
    model.add(Linear(256, 10, name='fclayer2',
                      initializer=Gaussian(std=0.01)))
    return model

mnist = MNIST()
mnist.load()
model = MyModel()
loss = SoftmaxCrossEntropy(num_class=10)

# define your learning rate sheduler
def func(lr, iteration):
    if iteration % 1000 ==0:
        return lr*0.5
    else:
        return lr

adam = Adam(lr=0.00095, decay=0,  sheduler_func=None, bias_correction=True)
l2 = L2(w=0.001) # L2 regularization with lambda=0.001
model.compile(optimizer=adam, loss=loss, regularization=l2)

import time
start = time.time()
train_results, val_results, test_results = model.train(
    mnist,
    train_batch=50, val_batch=1000, test_batch=1000,
    epochs=2,
    val_intervals=-1, test_intervals=900, print_intervals=100)
print('cost:', time.time()-start)