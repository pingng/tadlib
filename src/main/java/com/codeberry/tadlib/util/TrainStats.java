package com.codeberry.tadlib.util;

import com.codeberry.tadlib.nn.Model;
import com.codeberry.tadlib.tensor.Tensor;

import static com.codeberry.tadlib.util.AccuracyUtils.softmaxAccuracy;

public class TrainStats {
    private final long createdAt = System.currentTimeMillis();
    private double costTotal;
    private double costL2Total;

    private double accTotal;
    private int iterations;

    private double lastAccuracy;
    private double lastCost;

    public void accumulate(double cost, double l2Cost, double accuracy) {
        this.lastAccuracy = accuracy;
        this.lastCost = cost;

        costTotal += cost;
        costL2Total += l2Cost;
        accTotal += accuracy;
        iterations++;
    }

    public double getLastCost() {
        return lastCost;
    }

    public double getLastAccuracy() {
        return lastAccuracy;
    }

    @Override
    public String toString() {
        return "TrainStats{" +
                "iterations=" + iterations +
                ", accuracy=" + getAverageAccuracy() +
                ", costTotal=" + getAverageCost() +
                ", costL2Total=" + (costL2Total / iterations) +
                ", createdSince=" + ((System.currentTimeMillis() - createdAt) / 1000.0) +
                '}';
    }

    private double getAverageCost() {
        return costTotal / iterations;
    }

    public double getAverageAccuracy() {
        return accTotal / iterations;
    }

    public void accumulate(Model.PredictionAndLosses pl, Tensor labels) {
        accumulate(
                (double) pl.totalLoss.toDoubles(),
                (double) pl.l2Loss.toDoubles(),
                softmaxAccuracy(labels, pl.prediction));
    }

    public Model.OutputStats asOutputStats() {
        return new Model.OutputStats(getAverageCost(), getAverageAccuracy());
    }
}
