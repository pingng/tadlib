package com.codeberry.tadlib.mnist;

import com.codeberry.tadlib.array.TArrayFactory;
import com.codeberry.tadlib.provider.java.NDArray;
import com.codeberry.tadlib.tensor.Tensor;
import com.codeberry.tadlib.util.AccuracyUtils;
import com.codeberry.tadlib.util.Batch;
import com.codeberry.tadlib.util.TrainingData;

import java.util.Random;

import static com.codeberry.tadlib.mnist.MNISTLoader.IMAGE_SIZE;
import static com.codeberry.tadlib.mnist.MNISTLoader.OUTPUTS;
import static com.codeberry.tadlib.provider.ProviderStore.shape;
import static com.codeberry.tadlib.tensor.Ops.*;
import static com.codeberry.tadlib.tensor.Tensor.constant;
import static com.codeberry.tadlib.tensor.Tensor.tensor;
import static com.codeberry.tadlib.util.TrainingDataUtils.toOneHot;

class TrainHardCodedFullyConnectedMNISTModel {

    private final TrainingData trainingData;

    private final Tensor hiddenW;
    private final Tensor hiddenB;
    private final Tensor outW;
    private final Tensor outB;

    public TrainHardCodedFullyConnectedMNISTModel(TrainingData trainingData) {
        this.trainingData = trainingData;

        Random weightRnd = new Random(Config.SEED);

        hiddenW = tensor(TArrayFactory.randomWeight(weightRnd, shape(IMAGE_SIZE * IMAGE_SIZE, Config.HIDDEN_UNITS)));
        hiddenB = tensor(TArrayFactory.randomWeight(weightRnd, shape(Config.HIDDEN_UNITS)));

        outW = tensor(TArrayFactory.randomWeight(weightRnd, shape(Config.HIDDEN_UNITS, OUTPUTS)));
        outB = tensor(TArrayFactory.randomWeight(weightRnd, shape(OUTPUTS)));
    }

    public void trainForEpochs(int epochs) {
        int batchCount = trainingData.calcTrainingBatchCountOfSize(Config.BATCH_SIZE);

        for (int epoch = 0; epoch < epochs; epoch++) {
            train(epoch, batchCount);
//            test();
        }
    }

    double accuracy = 0, cost = Double.POSITIVE_INFINITY;

    private void train(int epoch, int batchCount) {


        SingleLinePrinter printer = new SingleLinePrinter();

        double accTotal = 0;
        double costTotal = 0;

        for (int batchId = 0; batchId < batchCount; batchId++) {
            Batch batch = trainingData.getTrainingBatch(batchId, Config.BATCH_SIZE);

            Tensor prediction = forward(batch.input);
            Tensor cost = calcCost(batch, prediction);

            cost.backward();

            updateWeights();

            double acc = AccuracyUtils.softmaxAccuracy(batch.output, prediction);
            accTotal += acc;

            costTotal += (double) cost.toDoubles();

            this.cost = (costTotal / (batchId + 1));
            this.accuracy = (accTotal / (batchId + 1));

            printer.print("Epoch " + epoch + " (" + (batchId * 100 / batchCount) + "%): cost=" +
                    this.cost + ", accuracy=" + this.accuracy);
        }

        printer.println("--- Epoch " + epoch + ": cost=" +
                (costTotal / batchCount) + ", accuracy=" + (accTotal / batchCount));
    }

    private static Tensor calcCost(Batch batch, Tensor prediction) {
        Tensor totalSoftmaxCost = sumSoftmaxCrossEntropy(toOneHot(batch.output, OUTPUTS), prediction);

        return div(totalSoftmaxCost, constant(batch.getBatchSize()));
    }

    private void updateWeights() {
        hiddenW.update(TrainHardCodedFullyConnectedMNISTModel::sgdUpdate);
        hiddenB.update(TrainHardCodedFullyConnectedMNISTModel::sgdUpdate);
        outW.update(TrainHardCodedFullyConnectedMNISTModel::sgdUpdate);
        outB.update(TrainHardCodedFullyConnectedMNISTModel::sgdUpdate);
    }

    private static NDArray sgdUpdate(NDArray currentValues, NDArray gradient) {
        return currentValues.sub(gradient.mul(Config.LEARNING_RATE));
    }

    private void test() {
        Tensor prediction = forward(trainingData.xTest);
        double testAccuracy = AccuracyUtils.softmaxAccuracy(trainingData.yTest, prediction);
        System.out.println("Test acc: " + testAccuracy);
    }

    private Tensor forward(Tensor input) {
        Tensor xTrain = reshape(input, -1, IMAGE_SIZE * IMAGE_SIZE);
        Tensor firstLayer = relu(add(
                matmul(xTrain, hiddenW),
                hiddenB));

        return add(matmul(firstLayer, outW), outB);
    }

    private static class Config {
        static final int SEED = 4;
        static final int HIDDEN_UNITS = 128;
        static final int BATCH_SIZE = 32;
        static final double LEARNING_RATE = 0.01;
    }

    private static class SingleLinePrinter {
        private String lastString = "";

        void print(String str) {
            printBackspaces(lastString.length());

            System.out.print(str);
            lastString = str;
        }

        void println(String str) {
            printBackspaces(lastString.length());

            System.out.println(str);
            lastString = "";
        }

        private static void printBackspaces(int count) {
            for (int i = 0; i < count; i++) {
                System.out.print("\b");
            }
        }
    }
}
