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

    public TrainingData getTrainingBatch(int batchIndex, int size) {
        return new TrainingData(xTrain.subBatch(batchIndex, size),
                yTrain.subBatch(batchIndex, size), xTest, yTest);
    }

    public int getTrainingBatchSize() {
        return xTrain.getShape().at(0);
    }
}
