package com.codeberry.tadlib.example.mnist;

import com.codeberry.tadlib.example.TrainingData;
import com.codeberry.tadlib.tensor.Tensor;

import java.util.Random;

import static com.codeberry.tadlib.array.Shape.shape;
import static com.codeberry.tadlib.array.TArray.randWeight;
import static com.codeberry.tadlib.example.mnist.MNISTLoader.toOneHot;
import static com.codeberry.tadlib.tensor.Ops.*;
import static com.codeberry.tadlib.tensor.Tensor.constant;
import static com.codeberry.tadlib.tensor.Tensor.tensor;

class MNISTFullyConnectedModel {

    private final Tensor hiddenW;
    private final Tensor hiddenB;
    private final Tensor outW;
    private final Tensor outB;

    MNISTFullyConnectedModel(Random weightRnd) {
        hiddenW = tensor(randWeight(weightRnd, shape(28 * 28, 32)));
        hiddenB = tensor(randWeight(weightRnd, shape(32)));

        outW = tensor(randWeight(weightRnd, shape(32, 10)));
        outB = tensor(randWeight(weightRnd, shape(10)));
    }

    // returns logits
    Tensor predict(Tensor input) {
        return forward(input);
    }

    Tensor trainSingleIteration(TrainingData batchData, double learningRate) {
        Tensor outLayer = forward(batchData.xTrain);

        Tensor cost = backpropGradient(batchData, outLayer);

        updateWeights(learningRate);

        return cost;
    }

    private Tensor forward(Tensor input) {
        Tensor xTrain = reshape(input, -1, 28 * 28);
        Tensor firstLayer = relu(add(
                matmul(xTrain, hiddenW),
                hiddenB));
        Tensor outLayer = add(
                matmul(firstLayer, outW),
                outB);
        return outLayer;
    }

    private static Tensor backpropGradient(TrainingData batchData, Tensor outLayer) {
        Tensor totalSoftmaxCost = sumSoftmaxCrossEntropy(toOneHot(batchData.yTrain), outLayer);
        Tensor avgSoftmaxCost = mul(totalSoftmaxCost, constant(1.0 / batchData.getTrainingBatchSize()));
        avgSoftmaxCost.backward();

        return avgSoftmaxCost;
    }

    private void updateWeights(double learningRate) {
        hiddenW.update((values, gradient) -> values.sub(gradient.mul(learningRate)));
        hiddenB.update((values, gradient) -> values.sub(gradient.mul(learningRate)));
        outW.update((values, gradient) -> values.sub(gradient.mul(learningRate)));
        outB.update((values, gradient) -> values.sub(gradient.mul(learningRate)));
    }
}
