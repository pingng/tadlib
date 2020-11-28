package com.codeberry.tadlib.example.mnist;

import com.codeberry.tadlib.example.TrainingData;
import com.codeberry.tadlib.tensor.Tensor;
import com.codeberry.tadlib.util.StringUtils;

import java.util.Random;

import static com.codeberry.tadlib.array.TArray.randWeight;
import static com.codeberry.tadlib.example.mnist.MNISTLoader.*;
import static com.codeberry.tadlib.example.mnist.MNISTLoader.LoadParams.params;
import static com.codeberry.tadlib.tensor.Tensor.tensor;
import static com.codeberry.tadlib.util.AccuracyUtils.softmaxAccuracy;

public class TrainSimpleMNISTMain {

    public static void main(String[] args) {
        TrainSimpleMNISTMain main = new TrainSimpleMNISTMain();

        main.trainModel(new TrainParams()
                .batchSize(32)
                .learningRate(0.05)
                .loaderParams(params()
                        .downloadWhenMissing(true)
                        .trainingExamples(40_000)
                        .testExamples(10_000)));
    }

    private void trainModel(TrainParams params) {
        System.out.println(StringUtils.toJson(params));

        MNISTFullyConnectedModel model = new MNISTFullyConnectedModel(new Random(4));

        TrainingData trainingData = load(params.loaderParams);
        int numberOfBatches = trainingData.calcTrainingBatchCountOfSize(params.batchSize);

        for (int epoch = 0; epoch <= 5000; epoch++) {
            System.out.println("=== Epoch " + epoch);
            for (int batchId = 0; batchId < numberOfBatches; batchId++) {
                TrainingData batchData = trainingData.getTrainingBatch(batchId, params.batchSize);

                Tensor loss = model.trainSingleIteration(batchData, params.learningRate);

                if (batchId % 200 == 0) {
                    System.out.println(batchId + "/" + numberOfBatches + ": Loss: " + loss.toDoubles());
                }
            }
            Tensor predict = model.predict(trainingData.xTest);
            double testAccuracy = softmaxAccuracy(trainingData.yTest, predict);
            System.out.println("* Test acc: " + testAccuracy);
        }
    }

    static class TrainParams {
        LoadParams loaderParams;
        double learningRate;
        int batchSize;

        TrainParams loaderParams(LoadParams loaderParams) {
            this.loaderParams = loaderParams;
            return this;
        }

        TrainParams batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        TrainParams learningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }
    }
}
