# Neural Networks & Pattern Recognition

This repo contains source code of all 4 homeworks and 1 project. (Still working on code organization).

## Homework 1 [Logistic & Softmax Regression via Gradient Descent](https://github.com/yrbszhsh/CSE-253/blob/Porj/Logistic%20and%20Softmax%20Regression%20via%20Gradient%20Descent.ipynb)

Applied two regression methods, i.e. logistic regression and softmax regression, to images of hand-written digits for classication as a procedure of digit recognition. Regression follows gradient descent direction. The outcome is evaluated by regulated cross-entropy in order to avoid over-fitting. 

In logistic regression, both datasets achieved correct rate over 95%. While in softmax regression, the whole dataset achieves correct rate at about 90%. In regularization, by optimizing parameters, regularization paremeter lambda is set to be 0.001 and regulation method is set to be L2.

<!---**Contributor: Pin Tian, Zhexi Zhang**--->

## Homework 2 [Multilayer Backpropagation Neural Networks](https://github.com/yrbszhsh/Neural-Networks-and-Pattern-Recoginition/blob/Porj/Multilayer%20Backpropagation%20Neural%20Networks.ipynb)

Built a multiple hidden layer neural network(Only one and two hidden layer used in this assignment), and applied some tricks on it. 

Basic neural network with sigmoid and softmax activation function achieved accuracy at 95%. Better activation function, tanh, and adaptive weights initialization helped accuracy increased to 98%. 

Shuffle and momentum has little help in accuracy but reduce calculating time by
10%.

In the experiment of ReLU accuracy has less than 1% increase and further reduced time.

<!---**Contributor: Pin Tian, Zhexi Zhang**--->

## Homework 3 [Deep Convolution Network & Transfer Learning](https://github.com/yrbszhsh/Neural-Networks-and-Pattern-Recoginition/blob/Porj/Deep%20Convolution%20Network%20%26%20Transfer%20Learning.ipynb)

Applied two method in image classification and compared influence of CNN models with different convolution depth. CNN is applied to images from CIFAR-10 dataset. 

Among all the experiments, selected the best CNN network with 8 convolution layer, 2 pooling layer and 3 fully connected layer which reaches 98.56% accuracy among training set and 90.24% among test set. Tricks such as image augment, batch normalization and weights initialization have been applied to the network.

<!---**Contributor: Pin Tian, Zhexi Zhang, Zhuoxi Zeng, Yuansheng Zhang**--->

## Homework 4 [Generating Music with Recurrent Networks](https://github.com/yrbszhsh/Neural-Networks-and-Pattern-Recoginition/blob/Porj/Generating%20Music%20with%20Recurrent%20Networks.ipynb)

Applied Recurrent Neural Networks (RNN) to ABC notation music generation task. Successfully generated several pieces of mucis using short 'prime' sequences.
Further studied the effects of some of the relevant parameters on the tranining effects by plotting the
evolutions of cross-entropy loss and prediction accuracy with epochs; the parameters include temperature, hidden layer size, learning rate, dropout rate and optimizer.

Evaluated the features of LSTM network by showing the heatmaps of activations of some typical neurons and found that the neurons can indeed extract features of music.
<!---**Contributor: Pin Tian, Zhexi Zhang, Zhuoxi Zeng, Yuansheng Zhang**--->

## Final Project [Image Caption Generation With Attention](https://github.com/yrbszhsh/Neural-Networks-and-Pattern-Recoginition/blob/Porj/Image%20Caption%20Generation%20With%20Attention.ipynb)

Optimized the published paper **[Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning](https://arxiv.org/pdf/1612.01887.pdf)** by implementing Bi-LSTM when generating caption. 

Evaluted result on recent published **[SPICE: Semantic Propositional Image Caption Evaluation](https://arxiv.org/pdf/1607.08822.pdf)** method and achieved 0.154 SPICE score.

Developed an image caption generation model by integrating the state-ofthe-art neural network models including pre-trained ResNet-152, adaptive attention network and novel bidirectional LSTM. Trained the Flickr8k dataset and evaluated the generated captions using SPICE and several other metrics. The adaptive attention model generated captions with CIDEr of 0.367 and SPICE of 0.113. Implementing novel bi-directional LSTM significantly improved the performances with the CIDEr increasing from 0.367 to 0.475, and SPICE increasing from 0.113 to 0.154. 

Further work still needs to be done to further improve the performances.


<!---**Contributor: Pin Tian, Zhexi Zhang, Zhuoxi Zeng, Yuansheng Zhang**--->
