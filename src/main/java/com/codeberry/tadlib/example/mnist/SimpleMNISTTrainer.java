package com.codeberry.tadlib.example.mnist;

import com.codeberry.tadlib.example.TrainingData;
import com.codeberry.tadlib.nn.model.Model;
import com.codeberry.tadlib.nn.model.ModelFactory;
import com.codeberry.tadlib.tensor.Tensor;
import com.codeberry.tadlib.util.StringUtils;

import java.util.Random;

import static com.codeberry.tadlib.example.mnist.MNISTConvModel.Config.Builder.cfgBuilder;
import static com.codeberry.tadlib.example.mnist.MNISTLoader.LoadParams.params;
import static com.codeberry.tadlib.util.AccuracyUtils.softmaxAccuracy;

public class SimpleMNISTTrainer {

    private final TrainParams params;
    private final TrainLogger logger = new TrainLogger();
    private final TrainingData trainingData;
    private final Model model;

    public SimpleMNISTTrainer(TrainParams params) {
        this.params = params;

        System.out.println(StringUtils.toJson(params));

        trainingData = MNISTLoader.load(params.loaderParams);
        model = params.modelFactory.createModel();
    }

    public void trainEpochs(int epochs) {
        int numberOfBatches = trainingData.calcTrainingBatchCountOfSize(params.batchSize);

        Random rnd = new Random(4);
        for (int epoch = 0; epoch <= epochs; epoch++) {
            System.out.println("=== Epoch " + epoch);
            MNISTConvModel.TrainStats stats = new MNISTConvModel.TrainStats();
            for (int batchId = 0; batchId < numberOfBatches; batchId++) {
                TrainingData batchData = trainingData.getTrainingBatch(batchId, params.batchSize);

                Model.PredictionAndLosses pl = model.trainSingleIteration(rnd, batchData, params.learningRate);

                stats.accumulate(pl, batchData.yTrain);

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

        public void log(int batchId, int numberOfBatches, Model model, MNISTConvModel.TrainStats stats) {
            int batchIdMod = batchId % OUTPUT_BATCHES;
            if (batchIdMod == 0) {
                batchProgress = 0;
                System.out.println("- Batch " + batchId + "/" + numberOfBatches);
                System.out.println("  " + stats);
                System.out.println(model.getTrainingLogText());
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
        ModelFactory modelFactory;
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

        TrainParams modelFactory(ModelFactory modelFactory) {
            this.modelFactory = modelFactory;
            return this;
        }
    }
}
