package com.codeberry.tadlib.example.mnist;

import com.codeberry.tadlib.example.TrainingData;
import com.codeberry.tadlib.tensor.Tensor;
import com.codeberry.tadlib.util.StringUtils;

import java.util.Random;

import static com.codeberry.tadlib.example.mnist.MNISTLoader.LoadParams.params;
import static com.codeberry.tadlib.example.mnist.MNISTConvModel.Config.Builder.cfgBuilder;
import static com.codeberry.tadlib.util.AccuracyUtils.softmaxAccuracy;

public class TrainMNISTMain {

    public static void main(String[] args) {
        TrainMNISTMain main = new TrainMNISTMain();

        main.trainModel(new TrainParams()
                .batchSize(32)
                .learningRate(0.15)
                .loaderParams(params()
                        .downloadWhenMissing(true)
                        .trainingExamples(40_000)
                        .testExamples(10_000))
                .modelConfig(cfgBuilder()
                        .firstConvChannels(4)
                        .secondConvChannels(8)
                        .fullyConnectedSize(32)
                        .l2Lambda(0.01)
                        .weightInitRandomSeed(4)
                        .useBatchNormalization(true)
                        .dropoutKeep(0.5)
                        .build()));
    }

    private void trainModel(TrainParams params) {
        System.out.println(StringUtils.toJson(params));

        TrainLogger logger = new TrainLogger();

        TrainingData trainingData = MNISTLoader.load(params.loaderParams);
        MNISTConvModel model = new MNISTConvModel(params.modelConfig);

        int numberOfBatches = trainingData.calcTrainingBatchCountOfSize(params.batchSize);

        Random rnd = new Random(4);
        for (int epoch = 0; epoch <= 5000; epoch++) {
            System.out.println("=== Epoch " + epoch);
            MNISTConvModel.TrainStats stats = new MNISTConvModel.TrainStats();
            for (int batchId = 0; batchId < numberOfBatches; batchId++) {
                TrainingData batchData = trainingData.getTrainingBatch(batchId, params.batchSize);

                model.trainSingleIteration(rnd, batchData, params.learningRate, stats);

                logger.log(batchId, numberOfBatches, model, stats);
            }
            System.out.println(stats);

            Tensor predict = model.predict(trainingData.xTest);
            double testAccuracy = softmaxAccuracy(trainingData.yTest, predict);
            System.out.println("* Test acc: " + testAccuracy);
        }
    }

    static class TrainLogger {
        static  final int OUTPUT_BATCHES = 200;
        int batchProgress = 0;

        public void log(int batchId, int numberOfBatches, MNISTConvModel model, MNISTConvModel.TrainStats stats) {
            int batchIdMod = batchId % OUTPUT_BATCHES;
            if (batchIdMod == 0) {
                batchProgress = 0;
                System.out.println("- Batch " + batchId + "/" + numberOfBatches);
                System.out.println("  " + stats);
                System.out.println("sec_bnAverages:\n" + model.sec_bnAverages);
                System.out.println("full_bnAverages:\n" + model.full_bnAverages);
            } else {
                int progress10Percent = batchIdMod * 10 / OUTPUT_BATCHES;
                if (progress10Percent != batchProgress) {
                    System.out.println(progress10Percent * 10 + "%");
                    batchProgress = progress10Percent;
                }
            }
        }
    }

    static class TrainParams {
        MNISTLoader.LoadParams loaderParams;
        MNISTConvModel.Config modelConfig;
        double learningRate;
        int batchSize;

        TrainParams loaderParams(MNISTLoader.LoadParams loaderParams) {
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

        TrainParams modelConfig(MNISTConvModel.Config modelConfig) {
            this.modelConfig = modelConfig;
            return this;
        }
    }
}
