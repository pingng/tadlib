package com.codeberry.tadlib.nn.model;

import com.codeberry.tadlib.tensor.Tensor;

import static com.codeberry.tadlib.util.AccuracyUtils.softmaxAccuracy;

public class TrainStats {
    private double costTotal;
    private double costL2Total;

    private double accTotal;
    private int iterations;

    public void accumulate(double cost, double l2Cost, double accuracy) {
        costTotal += cost;
        costL2Total += l2Cost;
        accTotal += accuracy;
        iterations++;
    }

    @Override
    public String toString() {
        return "TrainStats{" +
                "iterations=" + iterations +
                ", accuracy=" + (accTotal / iterations) +
                ", costTotal=" + (costTotal / iterations) +
                ", costL2Total=" + (costL2Total / iterations) +
                '}';
    }

    public void accumulate(Model.PredictionAndLosses pl, Tensor labels) {
        accumulate((double) pl.totalLoss.toDoubles(),
                (double) pl.l2Loss.toDoubles(),
                softmaxAccuracy(labels, pl.prediction));
    }
}
