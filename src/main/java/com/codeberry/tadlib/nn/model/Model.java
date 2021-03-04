package com.codeberry.tadlib.nn.model;

import com.codeberry.tadlib.array.NDArray;
import com.codeberry.tadlib.memorymanagement.DisposalRegister;
import com.codeberry.tadlib.example.TrainingData;
import com.codeberry.tadlib.nn.model.optimizer.Optimizer;
import com.codeberry.tadlib.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static com.codeberry.tadlib.provider.ProviderStore.array;
import static java.util.Collections.emptyList;

public interface Model {
    default PredictionAndLosses trainSingleIteration(Random rnd, TrainingData.Batch batchData, Optimizer optimizer, IterationInfo iterationInfo) {
        PredictionAndLosses l = calcGradient(rnd, batchData, iterationInfo);

        List<Tensor> params = getParams();

        optimizer.optimize(params);

        l.runTasks();

        return l;
    }

    default String getTrainingLogText() {
        return "";
    }

    Tensor predict(Tensor input, IterationInfo iterationInfo);

    PredictionAndLosses calcCost(Random rnd, TrainingData.Batch trainingData, IterationInfo iterationInfo);

    default PredictionAndLosses calcGradient(Random rnd, TrainingData.Batch trainingData, IterationInfo iterationInfo) {
        resetGradients();

        PredictionAndLosses l = calcCost(rnd, trainingData, iterationInfo);
        l.totalLoss.backward(array(1.0));

        return l;
    }

    default void resetGradients() {
        List<Tensor> params = getParams();
        for (Tensor p : params) {
            p.resetGradient();
        }
    }

    default List<NDArray> getGradients() {
        List<Tensor> params = getParams();

        List<NDArray> grads = new ArrayList<>();
        for (Tensor p : params) {
            grads.add(p.getGradient());
        }
        return grads;
    }

    List<Tensor> getParams();

    /**
     * @return Objects that are needed for the model to work and thus must not be disposed
     */
    default List<DisposalRegister.Disposable> getKeepInMemoryDisposables() {
        return emptyList();
    }

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

    class IterationInfo {
        public final int epoch;
        public final int batchIndex;
        public final int batchCount;
        public final TrainInfo prevEpochTrainInfo;

        public IterationInfo(int epoch, int batchIndex, int batchCount) {
            this(epoch, batchIndex, batchCount, null);
        }

        public IterationInfo(int epoch, int batchIndex, int batchCount, TrainInfo prevEpochTrainInfo) {
            this.epoch = epoch;
            this.batchIndex = batchIndex;
            this.batchCount = batchCount;
            this.prevEpochTrainInfo = prevEpochTrainInfo;
        }

        public boolean hasPrevEpochTrainInfo() {
            return prevEpochTrainInfo != null && prevEpochTrainInfo.training != null;
        }
    }

    class TrainInfo {
        public final OutputStats training;
        public final OutputStats test;

        public TrainInfo(OutputStats training, OutputStats test) {
            this.training = training;
            this.test = test;
        }

        public boolean isValid() {
            return training != null && test != null;
        }

        public boolean trainingIsMoreAccurateThanTesting() {
            return training.accuracy >= test.accuracy;
        }
    }

    class OutputStats {
        public final double cost;
        public final double accuracy;

        public OutputStats(double cost, double accuracy) {
            this.cost = cost;
            this.accuracy = accuracy;
        }
    }
}
