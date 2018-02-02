from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from scipy.signal import convolve2d

# Load Data
mndata = MNIST('mnist')
trainData, trainLabel = mndata.load_training()
testData, testLabel = mndata.load_testing()
trainData = np.asarray(trainData) / 127.5 - 1
testData  = np.asarray(testData) / 127.5 - 1
# Data Prepare
xTrain, yTrain = trainData[:60000], np.asarray(trainLabel)[:, np.newaxis]
xTest, yTest = testData, np.asarray(testLabel)[:, np.newaxis]

# One-hot on Labels:
def OneHot (Data, M):
    result = np.zeros((len(Data),M))
    for i in range(len(Data)):
        result[i,Data[i]] = 1
    return result

TrainSet = xTrain[:50000]
TrainLable = OneHot(yTrain[:50000], 10)
ValidSet = xTrain[-10000:]
ValidLabel = OneHot(yTrain[-10000:], 10)
TestSet = xTest[-2000:]
TestLabel = OneHot(yTest[-2000:], 10)

# Activate function
def logistFunc (x):
    return 1 / (1 + np.exp(-x))

def softFunc (x):
    return np.exp(x) / np.sum(np.exp(x), axis = 1)[:, np.newaxis]

# Error:
def softErr (t, y):
    Err = np.sum(np.multiply(t, np.log(y)))
    return Err

# Check if gradient descent:
# def check (w, x, t):
#     max_check = 5
#     flag = True
#     count = 0
#     epsilon = 1e-2
#     while (flag):
#         count += 1
#         y1 = softFunc(w + epsilon, x, k)
#         y2 = softFunc(w - epsilon, x, k)
#         Err_1 = (t, y1)
#         Err_2 = (t, y2)
#         gradE = (Err_1 - Err_2) / (2 * epsilon)
#
#         if gradE > 0:
#             flag = False
#             return False
#         elif count >= max_check:
#             flag = False
#             return True

# Neural Network:
def NN (x, w_ih, w_ho, num_node = 64):
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
    return y, zj, aj

def calculateLoss(data, label, weight_ih, weight_ho, lam):
    y = NN(data, weight_th, weight_ho)
    E = -1.0 * np.mean(label * np.log(y))
    # 根据求和计算方式，感觉可以对两个weight矩阵相加后再求和或者分别求和相加，结果应该是一样的
    C = np.sum(np.square(weight_ih) + np.square(weight_ho))
    loss = E + lam * C
    return loss

def softmaxRegression (trainData, trainLabel, lr = 1e-4, maxIter = 200, T = 2e8, lam = 1e-3, num_node = 64):
    classNum = trainLabel.shape[1]
    trainD, validD, trainL, validL = train_test_split(trainData, trainLabel, test_size = 0.1, random_state = 42)
    w_ih = np.zeros((len(trainData[0]), num_node))
    w_ho = np.zeros((num_node, classNum))
    it = 0
    while(it <= maxIter):
        lr1 = lr / (1.0 + it / T)
        y, z, a = NN(trainD, w_ih, w_ho)
        diff = trainL - y
        dE_ho = 1.0 * z.T.dot(diff) / z.shape[0]
        dC_ho = 2.0 * np.sum(w_ho)
        dJ_ho = dE_ho + lam * dC_ho
        w_ho += lr1 * dJ_ho
        G = np.multiply(logistFunc(a), 1 - logistFunc(a))
        #这里卷积的顺序调整了一下以便返回和(t-y)尺寸相同的矩阵，不然后面的尺寸对不上
        dk = convolve2d(diff, G, mode = 'same')
        dE_ih = 1.0 * trainD.T.dot(dk.dot(w_ih.T)) / trainD.shape[0]
        dC_ih = 2.0 * np.sum(w_ih)
        dJ_ih = dE_ih + lam * dC_ih
        w_ih += lr1 * dJ_ih
        lossV = calculateLoss(validD, validL, w_ih, w_ho, lam)
        if it % 100 == 0:
            print (lossV)
        it += 1
    print ('done')
    return w_ih, w_ho

weight_ih, weight_ho = softmaxRegression(TrainSet, TrainLable)
