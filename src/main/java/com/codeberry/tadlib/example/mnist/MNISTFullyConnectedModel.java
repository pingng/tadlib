package com.codeberry.tadlib.example.mnist;

import com.codeberry.tadlib.example.TrainingData;
import com.codeberry.tadlib.nn.model.Model;
import com.codeberry.tadlib.nn.model.ModelFactory;
import com.codeberry.tadlib.tensor.Tensor;

import java.util.Random;

import static com.codeberry.tadlib.array.Shape.shape;
import static com.codeberry.tadlib.array.TArray.randWeight;
import static com.codeberry.tadlib.example.mnist.MNISTLoader.*;
import static com.codeberry.tadlib.tensor.Ops.*;
import static com.codeberry.tadlib.tensor.Tensor.constant;
import static com.codeberry.tadlib.tensor.Tensor.tensor;

class MNISTFullyConnectedModel implements Model {

    private final Tensor hiddenW;
    private final Tensor hiddenB;
    private final Tensor outW;
    private final Tensor outB;

    public MNISTFullyConnectedModel(Config config) {
        Random weightRnd = new Random(config.weightInitRandomSeed);

        hiddenW = tensor(randWeight(weightRnd, shape(IMAGE_SIZE * IMAGE_SIZE, config.hiddenNeurons)));
        hiddenB = tensor(randWeight(weightRnd, shape(config.hiddenNeurons)));

        outW = tensor(randWeight(weightRnd, shape(config.hiddenNeurons, OUTPUTS)));
        outB = tensor(randWeight(weightRnd, shape(OUTPUTS)));
    }

    // returns logits
    @Override
    public Tensor predict(Tensor input) {
        return forward(input);
    }

    @Override
    public PredictionAndLosses trainSingleIteration(Random rnd, TrainingData batchData, double learningRate) {
        Tensor prediction = forward(batchData.xTrain);

        Tensor cost = backpropGradient(batchData, prediction);

        updateWeights(learningRate);

        return new PredictionAndLosses(prediction, cost);
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

    private static Tensor backpropGradient(TrainingData batchData, Tensor outLayer) {
        Tensor totalSoftmaxCost = sumSoftmaxCrossEntropy(toOneHot(batchData.yTrain), outLayer);
        Tensor avgSoftmaxCost = div(totalSoftmaxCost, constant(batchData.getTrainingBatchSize()));
        avgSoftmaxCost.backward();

        return avgSoftmaxCost;
    }

    private void updateWeights(double learningRate) {
        hiddenW.update((values, gradient) -> values.sub(gradient.mul(learningRate)));
        hiddenB.update((values, gradient) -> values.sub(gradient.mul(learningRate)));
        outW.update((values, gradient) -> values.sub(gradient.mul(learningRate)));
        outB.update((values, gradient) -> values.sub(gradient.mul(learningRate)));
    }

    public static class Config implements ModelFactory {
        private final int hiddenNeurons;
        private final long weightInitRandomSeed;

        private Config(int hiddenNeurons, long weightInitRandomSeed) {
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

            public static Builder cfgBuilder() {
                return new Builder();
            }

            public Config build() {
                return new Config(hiddenNeurons, weightInitRandomSeed);
            }
        }
    }
}
