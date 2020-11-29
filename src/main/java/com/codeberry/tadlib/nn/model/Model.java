package com.codeberry.tadlib.nn.model;

import com.codeberry.tadlib.example.TrainingData;
import com.codeberry.tadlib.example.mnist.MNISTConvModel;
import com.codeberry.tadlib.tensor.Tensor;

import java.util.Random;

public interface Model {
    PredictionAndLosses trainSingleIteration(Random rnd, TrainingData batchData, double learningRate);

    default String getTrainingLogText() {
        return "";
    }

    Tensor predict(Tensor input);

    class PredictionAndLosses {
        public final Tensor prediction;
        public final Tensor totalLoss;
        public final Tensor l2Loss;

        public PredictionAndLosses(Tensor prediction, Tensor totalCost) {
            this(prediction, totalCost, Tensor.ZERO);
        }

        public PredictionAndLosses(Tensor prediction, Tensor totalLoss, Tensor l2Loss) {
            this.prediction = prediction;
            this.totalLoss = totalLoss;
            this.l2Loss = l2Loss;
        }
    }

}
