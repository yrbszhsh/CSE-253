
# coding: utf-8

# # Load and Preprocess data

# In[1]:

# add bias term
def addBias(data):
    data = np.insert(data, data.shape[0], 1, axis=0)
    return data

def dataNorm(data, n_comp = 500):
    pca = PCA(n_comp)
    data_new = pca.fit_transform(data)
    return data_new
# activation function
#1. for hidden layer
def ReLU(X,W):
    a = np.dot(W.T, X)
    return a * (a > 0)
#2. for output layer
def softMax(X, W):
    ak = np.dot(W.T, X)
    scores = np.exp(ak)
    return scores / np.sum(scores, axis=0, keepdims=True)

def logistic(X, W):
    a = np.dot(W.T, X)
    return 1.0 / (1 + np.exp(-1.0 * a))

def tanh(X, W):
    a = np.dot(W.T, X)
    return 1.7159 * np.tanh((2/3) * a)

# randomly shuffle the data
def shuffle(X, t):
    ind = np.random.permutation(t.size)
    X, t = X[:, ind], t[ind]
    return X, t

# def plot_losses(method):
    fig, ax = plt.subplots()
    niter = method.loss.shape[1]
    x = np.linspace(1, niter / method.miniBatch, niter)
    ax.plot(x, method.loss[0], label="train loss")
    if method.earlyStop:
        ax.plot(x, method.loss[1], label="validation loss")
        ax.plot(x, method.loss[2], label="test loss")
    else:
        ax.plot(x, method.loss[1], label="test loss")
    ax.legend()
def plotLoss(Tar):
    loss = Tar.loss
    iteration = loss.shape[1]
    epacho = iteration / Tar.miniBatch
    xx = np.linspace(1, epacho, iteration)
    Flag = Tar.earlyStop
    plt.figure(size = (8, 6))
    plt.plot(xx, loss[0], label = 'train loss')
    plt.plot(xx, loss[-1], label = 'test loss')
    if loss.shape[0] == 3:
        plt.plot(xx, loss[1], label = 'validation loss')
    plt.legend()

def plotError(Tar):
    error = Tar.error
    iteration = loss.shape[1]
    epacho = iteration / Tar.miniBatch
    xx = np.linspace(1, epacho, iteration)
    Flag = Tar.earlyStop
    plt.figure(size = (8, 6))
    plt.plot(xx, loss[0], label = 'train percent correct')
    plt.plot(xx, loss[-1], label = 'test percent correct')
    if Flag:
        plt.plot(xx,error[1], label = 'validation percent correct')
    plt.legend()

# def plot_errors(method):
    fig, ax = plt.subplots()
    niter = method.error.shape[1]
    x = np.linspace(1, niter / method.miniBatch, niter)
    ax.plot(x, 1 - method.error[0], label="train percent correct")
    if method.earlyStop:
        ax.plot(x, 1 - method.error[1], label="validation percent correct")
        ax.plot(x, 1 - method.error[2], label="test percent correct")
    else:
        ax.plot(x, 1 - method.error[1], label="test percent correct")
    ax.legend(loc="lower right")


# In[4]:

from mnist import MNIST
from sklearn.decomposition import PCA
import math
import numpy as np
import time

t0 = time.clock()

mndata = MNIST('mnist')
trainData, trainLabel = mndata.load_training()
testData, testLabel = mndata.load_testing()



trainData = np.array(trainData).T / 255.
trainLabel = np.array(trainLabel).T
testData = np.array(testData).T / 255.
testLabel = np.array(testLabel).T

# zero mean and unit variance
trainData -= np.mean(trainData, axis=0)
trainData /= np.std(trainData, axis = 0)

testData -= np.mean(testData, axis=0)
testData /= np.std(testData, axis = 0)

trainData_pca = dataNorm(trainData,700)
testData_pca = dataNorm(testData,700)

testData_pca = addBias(testData_pca)
testData = addBias(testData)

# In[9]:

class multiLayer():
    def __init__(self, trainData, trainLabel, nHiddenLayer = (64,), lr = 1e-3, anneal = 0., maxIter = 1000, earlyStop=3, miniBatch = 1, lam = 0., momentum = 0.9, actFuns=("softMax","softMax")):
        self.X = trainData
        self.t = trainLabel
        # self.X = addBias(self.X)
        self.X = X
        self.earlyStop = earlyStop
        self.nClass = len(np.unique(trainLabel))
        if earlyStop:
            validSize = self.X.shape[1] / 6
            self.xValid = self.X[:,:validSize]
            self.tValid = self.t[:validSize]
            self.X = self.X[:,validSize:]
            self.t = self.t[validSize:]
        self.actFuns = actFuns
        self.lam = lam
        self.miniBatch = miniBatch
        self.loss = np.array([[], [], []]) if earlyStop else np.array([[], []])
        self.error = np.array([[], [], []]) if earlyStop else np.array([[], []])
        nLayer = len(nHiddenLayer) + 1
        self.numLayer = nLayer
        nUnits = (self.X.shape[0],) + nHiddenLayer + (self.nClass,)
        self.W = [1 / np.sqrt(nUnits[i]) * np.random.randn(nUnits[i] + 1, nUnits[i + 1]) for i in range(nLayer)]
        self.initW = self.W
        self.W = self.train(self.W, lr, maxIter, anneal, earlyStop, miniBatch, momentum)

    def train(self, weights, lr, maxIter, anneal, earlyStop, miniBatch, mu):
        d,n = self.X.shape
        batchSize = n / miniBatch
        self.wRecords = [[] for _ in weights]
        v = [0 for _ in weights]
        trainLoss = trainError = validLoss = validError = testLoss = testError = 0
        it = 0
        while it < maxIter:
            startInd , endInd = 0, batchSize
            self.X, self.t = shuffle(self.X, self.t)
            for i in range(miniBatch):
                xBatch, tBatch = self.X[:, startInd:endInd], self.t[startInd:endInd]

                if anneal:
                    if it * miniBatch + i > 5:
                        lr1 = lr / (1. + it / anneal)
                    else:
                        lr1 = 1e0
                else:
                    lr1 = lr

                outputs = self.forwardProp(xBatch, weights)
                dW = self.backProp(xBatch, tBatch, outputs, weights)
                trainLoss, trainError = self.evalLossError(xBatch, tBatch, weights)

                if i % 100 == 0:
                    testLoss, testError = self.evalTest(testData, testLabel, weights)

                for j in range(len(self.wRecords)):
                    self.wRecords[j].append(np.array(weights[j]))

                if earlyStop:
                    self.loss = np.hstack([self.loss, [[trainLoss], [validLoss], [testLoss]]])
                    self.error = np.hstack([self.error, [[trainError], [validError], [testError]]])
                    if i % 100 == 0:
                        validLoss, validError = self.evalLossError(self.xValid, self.tValid, weights)
                else:
                    self.loss = np.hstack([self.loss, [[trainLoss], [testLoss]]])
                    self.error = np.hstack([self.error, [[trainError], [testError]]])

                for j in range(len(weights)):
                    vPrev = v[j]
                    v[j] = mu * v[j] - lr * dW[j]
                    weights[j] += -mu * vPrev + (1 + mu) * v[j]

                startInd, endInd = startInd + batchSize, endInd + batchSize if i != miniBatch - 1 else n

            if earlyStop:
                print (trainLoss, trainError, validLoss, validError, testLoss, testError)
                if it != 0 and self.error[1, -1] >= self.error[1, -1 - miniBatch]:
                    stopCondition += 1
                    if stopCondition == earlyStop:
                        ind = self.error[2, :].argmin()
                        return [wRecord[ind] for wRecord in self.wRecords]
                else:
                    stopCondition = 0
            else:
                print (trainLoss, trainError, testLoss, testError)
            it += 1
        if earlyStop:
            ind = self.error[2, :].argmin()
            weights = [wRecord[ind] for wRecord in self.wRecords]
        return weights

    def evalLossError(self, X, t, weights):
        X = addBias(X)
        outputs = self.forwardProp(X, weights)
        y = outputs[-1]
        n = t.size
        loss = -np.sum(np.log(y[t, range(n)])) / n

        for weight in weights:
            loss += (np.sum(weight * weight)) * self.lam

        prediction = y.argmax(axis = 0)
        error = np.mean(prediction != t)
        return loss, error

    # def forwardProp(self, X, weights):
        outputs = [X]
        for weight, actFun in zip(weights, self.actFuns):
            if actFun == "softMax":
                outputs.append(softMax(outputs[-1], weight))
            elif actFun == "ReLU":
                outputs.append(ReLU(outputs[-1], weight))
            elif actFun == "logistic":
                outputs.append(logistic(outputs[-1], weight))
            elif actFun == "tanh":
                outputs.append(tanh(outputs[-1], weight))
        return outputs[1:]
    def forwardProp(self, X, weights):
        X = addBias(X)
        inputs = [X]
        outputs = []
        print(self.actFuns)
        for weight, actFun in zip(weights, self.actFuns):
            if actFun == "softMax":
                outputs.append(softMax(inputs[-1], weight))
                inputs.append(addBias(outputs[-1]))
                print(len(outputs))
            elif actFun == "ReLU":
                outputs.append(ReLU(inputs[-1], weight))
                inputs.append(addBias(outputs[-1]))
            elif actFun == "tanh":
                outputs.append(tanh(inputs[-1], weight))
                inputs.append(addBias(outputs[-1]))
            # print(len(outputs))
        return outputs

    # def backProp(self, X, t, outputs, weights):
        n = t.size
        dW = []
        outputs = [X] + outputs
        for i in range(len(weights) - 1, -1, -1):
            output1, output2 = outputs[i], outputs[i + 1]
            if self.actFuns[i] == "softMax":
                output2[t, range(n)] -= 1
                delta = output2 / n
                dW.append(np.dot(output1, delta.T))
            elif self.actFuns[i] == "ReLU":
                delta = (1. * (output2 > 0)) * np.dot(weights[i + 1], delta)
                dW.append(np.dot(output1, delta.T))
            elif self.actFuns[i] == "logistic":
                output2[t, range(n)] -= 1
                delta = output2 / n
                dW.append(np.dot(output1, delta.T))
            elif self.actFuns[i] == "tanh":
                G = output2 / 1.7159
                dG = (1 - np.power(G, 2)) * (2/3) * 1.7159
                delta = dG * np.dot(weights[i+1], delta)
                dW.append(np.dot(output1, delta.T))

        dW = dW[::-1]

        for i in range(len(weights)):
            dW[i] += 2 * weights[i] * self.lam
        return dW
    def backProp(self, X, t, outputs, weights):
        X = addBias(X)
        n = t.size
        dW = []
        outputs = [X] + outputs
        for i in range(len(weights) - 1, -1, -1):
            output1, output2 = outputs[i], outputs[i + 1]
            if self.actFuns[i] == "softMax":
                output2[t, range(n)] -= 1
                delta = output2 / n
                dW.append(np.dot(addBias(output1), delta.T))
                print(np.dot(addBias(output1), delta.T).shape)
            elif self.actFuns[i] == "ReLU":
                delta = (1. * (output2 > 0)) * np.dot(weights[i + 1], delta)
                dW.append(np.dot(output1, delta.T))
            elif self.actFuns[i] == "tanh":
                dG = (1 - np.power(output2 / 1.7159, 2)) * (2/3) * 1.7159
                delta = dG * np.dot(weights[i+1], delta)[1:]
                print(delta.shape)
                dW.append(np.dot(output1, delta.T))
                print(np.dot(output1, delta.T).shape)
        dW = dW[::-1]
        for i in range(len(weights)):
            dW[i] += 2 * weights[i] * self.lam
        return dW

    def evalTest(self, xTest, tTest, weights = 0):
        if type(weights) is int:
            return self.evalLossError(xTest, tTest, self.W)
        else:
            return self.evalLossError(xTest, tTest, weights)

# In[10]:
multilayer = multiLayer(trainData, trainLabel, lr = 1e-1, earlyStop = False, maxIter = 100, actFuns=("tanh", "softMax"), lam = 0., miniBatch = 128, nHiddenLayer = (64, ))


import matplotlib.pyplot as plt
test_loss, test_error = multilayer.evalTest(testData, testLabel)
print (1 - test_error)
t1 = time.clock() - t0
print ('Running Time =', t1)
# multilayer.plot_losses()
# multilayer.plot_errors()
plotLoss(multilayer)
plotError(multilayer)
plt.show()


# In[ ]:
