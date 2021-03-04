TADLib - Tiny Automatic Differentiation Library
===

What is TADLib?
---
TADLib was created for **understanding how ['autograd'](https://en.wikipedia.org/wiki/Automatic_differentiation) and basic neural networks can be implemented**. It provides 
Automatic Differentiation (AD) like Tensorflow and PyTorch, but is implemented in pure Java. Hardware acceleration 
is supported using [OpenCL](https://www.khronos.org/opencl/).

The main code uses tensors/multi dim arrays. A separate auto grad implementation using only scalar values can
also be found. It should be even easier to understand the concept of auto grad from this package.

Examples
---
### A fully connected Neural Network for the MNIST dataset
First we need some weights:
```java
hiddenW = tensor(randWeight(weightRnd, shape(28 * 28, 32)));
hiddenB = tensor(randWeight(weightRnd, shape(32)));

outW = tensor(randWeight(weightRnd, shape(32, 10)));
outB = tensor(randWeight(weightRnd, shape(10)));
```
Then we need the forward pass:
```java
// relu(inputs @ hiddenW + hiddenB)
Tensor firstLayer = relu(add(
        matmul(xTrain, hiddenW),
        hiddenB));
// (firstLayer @ outW + outB)
Tensor prediction = add(
        matmul(firstLayer, outW),
        outB);
```
The _outLayer_ is the output [logits](https://stackoverflow.com/questions/34240703/what-is-logits-softmax-and-softmax-cross-entropy-with-logits)
for each output classes.

We must then calculate the loss (scaled by the number of examples in the batch):
```java
Tensor totalSoftmaxCost = sumSoftmaxCrossEntropy(toOneHot(trainingData.output, 10), prediction);
Tensor avgSoftmaxCost = div(totalSoftmaxCost, constant(trainingData.getBatchSize()));
```

Then we trigger backpropagation of the gradients:
```java
avgSoftmaxCost.backward();
```

And, finally, update the weights ([plain SGD](https://ruder.io/optimizing-gradient-descent/index.html#batchgradientdescent)):
```java
hiddenW.update((values, gradient) -> values.sub(gradient.mul(learningRate)));
hiddenB.update((values, gradient) -> values.sub(gradient.mul(learningRate)));
outW.update((values, gradient) -> values.sub(gradient.mul(learningRate)));
outB.update((values, gradient) -> values.sub(gradient.mul(learningRate)));
```

#### Examples
- [TrainHardCodedFullyConnectedMNISTModel](src/main/java/com/codeberry/tadlib/example/mnist/TrainHardCodedFullyConnectedMNISTModel.java) \
  A simple & hard coded neural network, implemented in a single class.
- [TrainFixedConvMNISTMain](src/main/java/com/codeberry/tadlib/example/mnist/TrainFixedConvMNISTMain.java) \
  A neural network with convolution operations.
- [TrainConfiguredConvMNISTMain](src/main/java/com/codeberry/tadlib/example/mnist/TrainConfiguredConvMNISTMain.java) \
  Another convolutional neural network. It is built using layers.
- [TrainFixedMNISTConvAttentionMain](src/main/java/com/codeberry/tadlib/example/mnist/TrainFixedMNISTConvAttentionMain.java) \
  Trains an experimental with self attention.

OpenCL support
===
OpenCL support can be enabled by assigning the provider:
```java
ProviserStore.setProvider(new OpenCLProvider());
```
Operations will run **a lot** faster using OpenCL.

The OpenCL integration, as the java code, is naive & minimal to allow for (hopefully) better readability.
The performance will certainly not reach the level of Tensorflow nor PyTorch,
but it will be fast enough for more experimentation and less time waiting.

See the [opencl package](src/main/java/com/codeberry/tadlib/provider/opencl/README.md) for more details.

Scalar implementation
===
An even simpler implementation using scalar values can be found in the [singlevalue package](src/main/java/com/codeberry/tadlib/singlevalue/README.md).

About
---
### What is the point/goal of TADLib?
The focus of TADLib is to show how nn works under the hood. It runs conceptually like
Tensorflow or PyTorch in eager/immediate mode. TADLib is of course much more simple and 
runs orders of magnitude slower. The advantage is that it allows you to follow/debug/trace
the flow of each value, since it is implemented with plain double arrays and uses
normal java math operations.

The code is meant to be simple to read and not too difficult to follow. Some limitations are:
- limited set of math ops
- no/minimal optimizations
- immutable (mostly)
- ...which means it is slow :)

### What can it do?
It provides all the primitives to implement a standard multi layered convolutional neural net
for the MNIST-class problems. Using TADLib is like coding a nn in Tensorflow using Variables and
math ops to manually create the layers and structure of the model, but with the added verbosity of Java.

It is possible to create larger models with TADLib, but it will run too slow to be practically usable.

There is support for parallel execution of some math operations. It helps training of MNIST like
dataset, but it will still be too slow for real world problems.

References
---
The main auto grad structure of TADLib is heavily inspired by Joel Grus' auto grad tutorial:

    https://www.youtube.com/playlist?list=PLeDtc0GP5ICldMkRg-DkhpFX1rRBNHTCs
    (Livecoding an Autograd Library)

A huge thanks to Joel :)

Other refs:
- the Keras source code
- Andrew Ng's ML tutorials
- https://gombru.github.io/2018/05/23/cross_entropy_loss/
- https://wiseodd.github.io/techblog/2016/07/04/batchnorm/
- https://youtu.be/pHMzNW8Agq4 \
  ("Neural Networks Demystified \[Part 5: Numerical Gradient Checking]")
- https://towardsdatascience.com/understanding-the-scaling-of-l%C2%B2-regularization-in-the-context-of-neural-networks-e3d25f8b50db
- https://medium.com/@2017csm1006/forward-and-backpropagation-in-convolutional-neural-network-4dfa96d7b37e
- https://aerinykim.medium.com/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d
- https://youtube.com/playlist?list=PLIXJ-Sacf8u60G1TwcznBmK6rEL3gmZmV
  (self attention)
- https://papers.nips.cc/paper/2018/file/7edcfb2d8f6a659ef4cd1e6c9b6d7079-Paper.pdf
  (block drop)
  
License
---
Copyright © 2021, [Ping Ng](https://github.com/pingng)
Released under the [MIT License](LICENSE.txt).