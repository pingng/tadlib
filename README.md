TADLib - Tiny Automatic Differentiation Library
===

TODO: REWRITE TO ABOUT SUPPORT FOR OpenCL!!!

What is TADLib?
---
TADLib is a simple library for **understanding how neural networks works**. It provides 
Automatic Differentiation (AD) like Tensorflow and PyTorch, but is implemented in pure Java.

Examples
---
### A fully connected NN for MNIST
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
Tensor outLayer = add(
        matmul(firstLayer, outW),
        outB);
```
The _outLayer_ is the output, [logits](https://stackoverflow.com/questions/34240703/what-is-logits-softmax-and-softmax-cross-entropy-with-logits)
for each of the output classes.

We must then calculate the loss (scaled by the number of examples in the batch):
```java
Tensor totalSoftmaxCost = sumSoftmaxCrossEntropy(toOneHot(batchData.yTrain), outLayer);
Tensor avgSoftmaxCost = div(totalSoftmaxCost, constant(actualBatchSize));
```

Then trigger backpropagation of the gradients:
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

See [the source](src/main/java/com/codeberry/tadlib/example/mnist/TrainFullyConnectedMNISTMain.java) and
the [model](src/main/java/com/codeberry/tadlib/example/mnist/MNISTFullyConnectedModel.java) for this example.

There is also [another example](src/main/java/com/codeberry/tadlib/example/mnist/TrainFixedConvMNISTMain.java) that uses
[convolutions with batch normalization](src/main/java/com/codeberry/tadlib/example/mnist/FixedMNISTConvModel.java).
This is a hardcoded model.

[TrainConfiguredConvMNISTMain](src/main/java/com/codeberry/tadlib/example/mnist/TrainConfiguredConvMNISTMain.java)
is an example of a conv model built/configured in runtime using layers.

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

License
---
Copyright © 2020, [Ping Ng](https://github.com/pingng)
Released under the [MIT License](LICENSE.txt).