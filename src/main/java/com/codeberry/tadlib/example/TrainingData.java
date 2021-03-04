package com.codeberry.tadlib.example;

import com.codeberry.tadlib.tensor.Tensor;

public class TrainingData {
    public final Tensor xTrain;
    public final Tensor yTrain;
    public final Tensor xTest;
    public final Tensor yTest;

    public TrainingData(Tensor xTrain, Tensor yTrain, Tensor xTest, Tensor yTest) {
        this.xTrain = xTrain;
        this.yTrain = yTrain;
        this.xTest = xTest;
        this.yTest = yTest;
    }

    public int calcTrainingBatchCountOfSize(int batchSize) {
        int exCount = xTrain.getShape().at(0);
        return (exCount + batchSize - 1) / batchSize;
    }

    public int calcTestBatchCountOfSize(int batchSize) {
        int exCount = xTest.getShape().at(0);
        return (exCount + batchSize - 1) / batchSize;
    }

    public Batch getTrainingBatch(int batchIndex, int size) {
        return new Batch(xTrain.subBatch(batchIndex, size),
                yTrain.subBatch(batchIndex, size));
    }

    public Batch getTrainingBatchAll() {
        int examples = xTrain.getShape().at(0);
        return getTrainingBatch(0, examples);
    }

    public Batch getTestBatch(int batchIndex, int size) {
        return new Batch(xTest.subBatch(batchIndex, size),
                yTest.subBatch(batchIndex, size));
    }

    public static class Batch {
        public final Tensor input;
        public final Tensor output;

        public Batch(Tensor input, Tensor output) {
            this.input = input;
            this.output = output;
        }

        public int getBatchSize() {
            return input.getShape().at(0);
        }
    }
}
