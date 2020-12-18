package com.codeberry.tadlib.nn.model;

import com.codeberry.tadlib.array.JavaArray;
import com.codeberry.tadlib.example.TrainingData;
import com.codeberry.tadlib.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static com.codeberry.tadlib.array.TArrayFactory.value;

public interface Model {
    default PredictionAndLosses trainSingleIteration(Random rnd, TrainingData batchData, Optimizer optimizer) {
        PredictionAndLosses l = calcGradient(rnd, batchData);

        List<Tensor> params = getParams();

        optimizer.optimize(params);

        l.runTasks();

        return l;
    }

    default String getTrainingLogText() {
        return "";
    }

    Tensor predict(Tensor input);

    PredictionAndLosses calcCost(Random rnd, TrainingData trainingData);

    default PredictionAndLosses calcGradient(Random rnd, TrainingData trainingData) {
        resetGradients();

        PredictionAndLosses l = calcCost(rnd, trainingData);
        l.totalLoss.backward(value(1.0));

        return l;
    }

    private void resetGradients() {
        List<Tensor> params = getParams();
        for (Tensor p : params) {
            p.resetGradient();
        }
    }

    default List<JavaArray> getGradients() {
        List<Tensor> params = getParams();

        List<JavaArray> grads = new ArrayList<>();
        for (Tensor p : params) {
            grads.add(p.getGradient());
        }
        return grads;
    }

    List<Tensor> getParams();

    Model copy();

    class PredictionAndLosses {
        public final Tensor prediction;
        public final List<Runnable> trainingTasks;
        public final Tensor totalLoss;
        public final Tensor l2Loss;

        public PredictionAndLosses(Tensor prediction, List<Runnable> trainingTasks, Tensor totalCost) {
            this(prediction, trainingTasks, totalCost, Tensor.ZERO);
        }

        public PredictionAndLosses(Tensor prediction, List<Runnable> trainingTasks, Tensor totalLoss, Tensor l2Loss) {
            this.prediction = prediction;
            this.trainingTasks = trainingTasks;
            this.totalLoss = totalLoss;
            this.l2Loss = l2Loss;
        }

        private void runTasks() {
            for (Runnable tt : trainingTasks) {
                tt.run();
            }
        }
    }

}
