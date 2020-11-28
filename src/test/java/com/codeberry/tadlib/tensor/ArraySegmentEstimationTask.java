package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.example.mnist.MNISTConvModel;

import java.util.Random;
import java.util.concurrent.Callable;

class ArraySegmentEstimationTask implements Callable<double[]> {
    private final int rndSeed;
    private final MNISTConvModel model;
    private final int paramIndex;
    private final int fromIndex;
    private final int endIndex;
    private final Tensor xTrain;
    private final Tensor yTrain;

    public ArraySegmentEstimationTask(int rndSeed, MNISTConvModel model, int paramIndex, int fromIndex, int endIndex, Tensor xTrain, Tensor yTrain) {
        this.rndSeed = rndSeed;
        this.model = model;
        this.paramIndex = paramIndex;
        this.fromIndex = fromIndex;
        this.endIndex = endIndex;
        this.xTrain = xTrain;
        this.yTrain = yTrain;
    }

    @Override
    public double[] call() {
        double d1 = 0.000005;
        double d2 = 0.000001;

        double[] wData = model.getParam(paramIndex).getInternalData();
        double[] grad = new double[endIndex - fromIndex];

        for (int i = fromIndex; i < endIndex; i++) {
            if (i % 1000 == 0) {
                System.out.println(((i - fromIndex) * 100 / grad.length) + "%");
            }

            double org = wData[i];

            double estimate1 = estimateWithRespectToElement(d1, wData, i, org);

            grad[i - fromIndex] = estimate1;
//            double estimate1 = estimateWithRespectToElement(d1, wData, i, org);
//            double estimate2 = estimateWithRespectToElement(d2, wData, i, org);
//
//            grad[i - fromIndex] = (estimate1 + estimate2) * 0.5;

            wData[i] = org;
        }

        return grad;
    }

    private double estimateWithRespectToElement(double d, double[] wData, int elementIndex, double orgValue) {
        wData[elementIndex] = orgValue - d;
        double before = calcCost();
        wData[elementIndex] = orgValue + d;
        double after = calcCost();
        return (after - before) / (2 * d);
    }

    private double calcCost() {
        return (double) model.calcCost(new Random(rndSeed), xTrain, yTrain, new MNISTConvModel.TrainStats())
                .toDoubles();
    }

    @Override
    public String toString() {
        return " [" + fromIndex + "," + endIndex + ">";
    }
}
