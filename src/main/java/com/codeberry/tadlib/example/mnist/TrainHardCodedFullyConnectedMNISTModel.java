package com.codeberry.tadlib.example.mnist;

import com.codeberry.tadlib.array.NDArray;
import com.codeberry.tadlib.array.TArrayFactory;
import com.codeberry.tadlib.example.TrainingData;
import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.java.JavaProvider;
import com.codeberry.tadlib.tensor.Tensor;
import com.codeberry.tadlib.util.AccuracyUtils;

import java.util.Random;

import static com.codeberry.tadlib.example.mnist.MNISTLoader.IMAGE_SIZE;
import static com.codeberry.tadlib.example.mnist.MNISTLoader.LoadParams.params;
import static com.codeberry.tadlib.example.mnist.MNISTLoader.OUTPUTS;
import static com.codeberry.tadlib.provider.ProviderStore.shape;
import static com.codeberry.tadlib.tensor.Ops.*;
import static com.codeberry.tadlib.tensor.Ops.matmul;
import static com.codeberry.tadlib.tensor.Tensor.constant;
import static com.codeberry.tadlib.tensor.Tensor.tensor;
import static com.codeberry.tadlib.util.TrainingDataUtils.toOneHot;

public class TrainHardCodedFullyConnectedMNISTModel {
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

    public static void main(String[] args) {
//        ProviderStore.setProvider(new OpenCLProvider());
        ProviderStore.setProvider(new JavaProvider());

        TrainingData trainingData = MNISTLoader.load(params()
                .loadRegularMNIST()
                .downloadWhenMissing(true)
                .trainingExamples(40_000)
                .testExamples(10_000));

        TrainHardCodedFullyConnectedMNISTModel model = new TrainHardCodedFullyConnectedMNISTModel(trainingData);

        model.trainForEpochs(100);
    }

    private void trainForEpochs(int epochs) {
        int batchCount = trainingData.calcTrainingBatchCountOfSize(Config.BATCH_SIZE);

        for (int epoch = 0; epoch < epochs; epoch++) {
            train(epoch, batchCount);

            test();
        }
    }

    private void train(int epoch, int batchCount) {
        double accTotal = 0;
        double costTotal = 0;

        SingleLinePrinter printer = new SingleLinePrinter();

        for (int batchId = 0; batchId < batchCount; batchId++) {
            TrainingData.Batch batch = trainingData.getTrainingBatch(batchId, Config.BATCH_SIZE);

            Tensor prediction = forward(batch.input);
            Tensor cost = calcCost(batch, prediction);

            cost.backward();

            updateWeights();

            costTotal += (double) cost.toDoubles();
            accTotal += AccuracyUtils.softmaxAccuracy(batch.output, prediction);

            printer.print("Epoch " + epoch + " (" + (batchId * 100 / batchCount) + "%): cost=" +
                    (costTotal / (batchId + 1)) + ", accuracy=" + (accTotal / (batchId + 1)));
        }

        printer.println("--- Epoch " + epoch + ": cost=" +
                (costTotal / batchCount) + ", accuracy=" + (accTotal / batchCount));
    }

    private Tensor calcCost(TrainingData.Batch batch, Tensor prediction) {
        Tensor totalSoftmaxCost = sumSoftmaxCrossEntropy(toOneHot(batch.output, OUTPUTS), prediction);
        Tensor avgSoftmaxCost = div(totalSoftmaxCost, constant(batch.getBatchSize()));

        return avgSoftmaxCost;
    }

    private void updateWeights() {
        hiddenW.update(this::sgdUpdate);
        hiddenB.update(this::sgdUpdate);
        outW.update(this::sgdUpdate);
        outB.update(this::sgdUpdate);
    }

    private NDArray sgdUpdate(NDArray currentValues, NDArray gradient) {
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
