from mnist import MNIST
from sklearn.decomposition import PCA
import math
import numpy as np
import time

mndata = MNIST('mnist')
trainData, trainLabel = mndata.load_training()
testData, testLabel = mndata.load_testing()

xTrain, yTrain = trainData[:1000], trainLabel[:1000]
xTest, yTest = testData[-100:], testLabel[-100:]
# unified data
xTrain = np.array(xTrain).T / 255.
yTrain = np.array(yTrain).T
xTest = np.array(xTest).T / 255.
yTest = np.array(yTest).T
# zero mean and unit variance
xTrain -= np.mean(xTrain, axis=0)
xTrain /= np.std(xTrain, axis = 0)
xTest -= np.mean(xTest, axis=0)
xTest /= np.std(xTest, axis = 0)

def addBias(data):
    data = np.insert(data, data.shape[0], 1, axis=0)
    return data
# Activation Function
def ReLU(X,W):
    a = np.dot(W.T, X)
    return a * (a > 0)

def softMax(X, W):
    ak = np.dot(W.T, X)
    scores = np.exp(ak)
    return scores / np.sum(scores, axis=0, keepdims=True)

def tanh(X, W):
    a = np.dot(W.T, X)
    return 1.7159 * np.tanh((2/3) * a)
# Forward propagation
def forwardProp(X, weights, actFncs):
    X = addBias(X)
    inputs = [X]
    outputs = []
    for weight, actFun in zip(weights, actFncs):
        if actFun == "softMax":
            outputs.append(softMax(inputs[-1], weight))
            inputs.append(addBias(outputs[-1]))
        elif actFun == "ReLU":
            outputs.append(softMax(inputs[-1], weight))
            inputs.append(addBias(outputs[-1]))
        elif actFun == "tanh":
            outputs.append(softMax(inputs[-1], weight))
            inputs.append(addBias(outputs[-1]))
    return outputs
# Back prapagation
def backProp(X, t, outputs, weights, actFncs, lam):
    X = addBias(X)
    n = t.size
    dW = []
    outputs = [X] + outputs
    for i in range(len(weights) - 1, -1, -1):
        output1, output2 = outputs[i], outputs[i + 1]
        if actFncs[i] == "softMax":
            output2[t, range(n)] -= 1
            delta = output2 / n
            dW.append(np.dot(addBias(output1), delta.T))
            print(np.dot(addBias(output1), delta.T).shape)
        elif actFncs[i] == "ReLU":
            delta = (1. * (output2 > 0)) * np.dot(weights[i + 1], delta)
            dW.append(np.dot(output1, delta.T))
        elif actFncs[i] == "tanh":
            G = output2 / 1.7159
            dG = (1 - np.power(G, 2)) * (2/3) * 1.7159
            delta = dG * np.dot(weights[i+1], delta)[1:]
            print(delta.shape)
            dW.append(np.dot(output1, delta.T))
            print(np.dot(output1, delta.T).shape)
    dW = dW[::-1]
    for i in range(len(weights)):
        dW[i] += 2 * weights[i] * lam
    return dW
# Cross-entropy
def crossEntropy(X, t, weights, actFncs, lam):
    predY = forwardProp(X, weights, actFncs)
    Y = predY[-1]
    n = t.size
    loss = -np.sum(np.log(Y[t, range(n)])) / n
    for weight in weights:
        loss += lam * np.sum(np.square(weight))
    return loss
# Numerical gradient
def numericGrad(X, t, weights, epsilon, actFncs, lam):
    Grad = []
    for l in range(len(weights)):
        xx, yy = weights[l].shape
        numGrad = np.zeros((xx, yy))
        xAxis, yAxis = np.meshgrid(range(xx), range(yy))
        for i,j in zip(xAxis, yAxis):
            adjweight_1 = adjweight_2 = weights
            adjweight_1[l][i, j] += epsilon
            adjweight_2[l][i, j] -= epsilon
            entropy1 = crossEntropy(X, t, adjweight_1, actFncs, lam)
            entropy2 = crossEntropy(X, t, adjweight_2, actFncs, lam)
            numGrad[i ,j] = (entropy1 - entropy2) / (2 * epsilon)
        Grad.append(numGrad)
    return Grad
# Check gradient difference
def checkGradient(X, t, actFncs, layers, epsilon, lam = 1e-3):
    nLayer = len(layers) + 1
    nClass = len(np.unique(t))
    nHiddenLayers = ()
    # for i in range(len(layers)):
    #     nHiddenLayers += (layers[i] + 1, )
    nFanin = (X.shape[0], ) + layers + (nClass, )
    wInit = [1 / np.sqrt(nFanin[i]) * np.random.randn(nFanin[i]+1, nFanin[i+1]) for i in range(nLayer)]
    print(wInit[0].shape, wInit[1].shape)
    hiddenOut = forwardProp(X, wInit, actFncs)
    print(len(hiddenOut))
    trueGradient = backProp(X, t, hiddenOut, wInit, actFncs, lam)
    numGrandient = numericGrad(X, t, wInit, epsilon, actFncs, lam)
    diff = 0
    for k in range(len(trueGradient)):
        diff += np.abs(np.mean(trueGradient[k] - numGrandient[k]))
    return diff

d = checkGradient(xTrain, yTrain, ('tanh', 'softMax'), (64, ), 1e-2)
print(d)
