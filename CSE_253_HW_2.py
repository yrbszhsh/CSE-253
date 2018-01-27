from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Load Data
mndata = MNIST('mnist')
trainData, trainLabel = np.asarray(mndata.load_training())
testData, testLabel = np.asarray(mndata.load_testing())

# Data Prepare
xTrain, yTrain = trainData[:60000]/127.5 - 1, trainLabel
xTest, yTest = testData/127.5 - 1, testData

# One-hot on Labels:
def OneHot (Data, M):
    result = np.zeros((len(Data),M))
    for i in range(len(Dat)):
        result[i,Data[i]] = 1
    return result
TrainSet = xTrain[:50000]
TrainLable = OneHot(yTrain[:50000], 10)
ValidSet = xTrain[-10000:]
ValidLabel = OneHot(yTrain[-10000:], 10)
TestSet = xTest[-2000:]
TestLabel = OneHot(yTest[-2000:], 10)

# Activate function
def logistFunc (w, x):
    a = np.dot(np.transpose(w), x)
    return 1 / (1 + np.exp(-a))

def softFunc (x):
    return np.exp(x) / np.sum(np.exp(a), axis = 1)[:, np.newaxis]

# Error:
def softErr (t, y):
    Err = np.sum(np.multiply(t, np.log(y)))
    return Err

# Check if gradient descent:
def check (w, x, t):
    max_check = 5
    flag = True
    count = 0
    epsilon = 1e-2
    while (flag):
        count += 1
        y1 = softFunc(w + epsilon, x, k)
        y2 = softFunc(w - epsilon, x, k)
        Err_1 = (t, y1)
        Err_2 = (t, y2)
        gradE = (Err_1 - Err_2) / (2 * epsilon)

        if gradE > 0:
            flag = False
            return False
        elif count >= max_check:
            flag = False
            return True

# Neural Network:
def NN (x, num_node = 64, w_ih, w_ho):
    # w_ih: d * num_node
    # w_ho:  node * 10
    # x: n * d
    # d - dimension of input matrix
    # n - number of examples
    # num_node: node number of hidden layer
    aj = np.dot(x, w_ih)
    zj = logistFunc(aj)
    ak = np.dot(zj, w_ho)
    y = softFunc(ak)
    return y
