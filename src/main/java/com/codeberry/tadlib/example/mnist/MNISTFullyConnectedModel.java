package com.codeberry.tadlib.example.mnist;

import com.codeberry.tadlib.example.TrainingData;
import com.codeberry.tadlib.nn.model.Model;
import com.codeberry.tadlib.nn.model.ModelFactory;
import com.codeberry.tadlib.tensor.Tensor;
import com.codeberry.tadlib.util.ReflectionUtils;

import java.util.List;
import java.util.Random;

import static com.codeberry.tadlib.array.Shape.shape;
import static com.codeberry.tadlib.array.TArray.randWeight;
import static com.codeberry.tadlib.example.mnist.MNISTLoader.*;
import static com.codeberry.tadlib.tensor.Ops.*;
import static com.codeberry.tadlib.tensor.Tensor.constant;
import static com.codeberry.tadlib.tensor.Tensor.tensor;
import static java.util.Arrays.asList;
import static java.util.Collections.emptyList;

public class MNISTFullyConnectedModel implements Model {

    private final Factory factory;

    private final Tensor hiddenW;
    private final Tensor hiddenB;
    private final Tensor outW;
    private final Tensor outB;

    public MNISTFullyConnectedModel(Factory factory) {
        this.factory = factory;
        Random weightRnd = new Random(factory.weightInitRandomSeed);

        hiddenW = tensor(randWeight(weightRnd, shape(IMAGE_SIZE * IMAGE_SIZE, factory.hiddenNeurons)));
        hiddenB = tensor(randWeight(weightRnd, shape(factory.hiddenNeurons)));

        outW = tensor(randWeight(weightRnd, shape(factory.hiddenNeurons, OUTPUTS)));
        outB = tensor(randWeight(weightRnd, shape(OUTPUTS)));
    }

    public MNISTFullyConnectedModel(MNISTFullyConnectedModel src) {
        this(src.factory);

        ReflectionUtils.copyFieldOfClass(Tensor.class,
                src, this,
                Tensor::copy);
    }

    // returns logits
    @Override
    public Tensor predict(Tensor input) {
        return forward(input);
    }

    @Override
    public PredictionAndLosses trainSingleIteration(Random rnd, TrainingData batchData, double learningRate) {
        PredictionAndLosses pl = calcCost(rnd, batchData);

        updateWeights(learningRate);

        return pl;
    }

    @Override
    public PredictionAndLosses calcCost(Random rnd, TrainingData trainingData) {
        Tensor prediction = forward(trainingData.xTrain);

        Tensor totalSoftmaxCost = sumSoftmaxCrossEntropy(toOneHot(trainingData.yTrain), prediction);
        Tensor avgSoftmaxCost = div(totalSoftmaxCost, constant(trainingData.getTrainingBatchSize()));

        return new PredictionAndLosses(prediction, emptyList(), avgSoftmaxCost);
    }

    private Tensor forward(Tensor input) {
        Tensor xTrain = reshape(input, -1, IMAGE_SIZE * IMAGE_SIZE);
        Tensor firstLayer = relu(add(
                matmul(xTrain, hiddenW),
                hiddenB));
        Tensor outLayer = add(
                matmul(firstLayer, outW),
                outB);
        return outLayer;
    }

    private void updateWeights(double learningRate) {
        hiddenW.update((values, gradient) -> values.sub(gradient.mul(learningRate)));
        hiddenB.update((values, gradient) -> values.sub(gradient.mul(learningRate)));
        outW.update((values, gradient) -> values.sub(gradient.mul(learningRate)));
        outB.update((values, gradient) -> values.sub(gradient.mul(learningRate)));
    }

    @Override
    public List<Tensor> getParams() {
        return asList(hiddenW, hiddenB, outW, outB);
    }

    @Override
    public Model copy() {
        return new MNISTFullyConnectedModel(this);
    }

    public static class Factory implements ModelFactory {
        private final int hiddenNeurons;
        private final long weightInitRandomSeed;

        private Factory(int hiddenNeurons, long weightInitRandomSeed) {
            this.hiddenNeurons = hiddenNeurons;
            this.weightInitRandomSeed = weightInitRandomSeed;
        }

        @Override
        public Model createModel() {
            return new MNISTFullyConnectedModel(this);
        }

        public static class Builder {
            private int hiddenNeurons;
            private long weightInitRandomSeed;

            public Builder hiddenNeurons(int hiddenNeurons) {
                this.hiddenNeurons = hiddenNeurons;
                return this;
            }

            public Builder weightInitRandomSeed(long weightInitRandomSeed) {
                this.weightInitRandomSeed = weightInitRandomSeed;
                return this;
            }

            public static Builder factoryBuilder() {
                return new Builder();
            }

            public Factory build() {
                return new Factory(hiddenNeurons, weightInitRandomSeed);
            }
        }
    }
}
