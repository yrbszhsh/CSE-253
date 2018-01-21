from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

mndata = MNIST('mnist')
trainData, trainLabel = mndata.load_training()
testData, testLabel = mndata.load_testing()
