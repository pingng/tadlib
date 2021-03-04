package com.codeberry.tadlib.nn.model.optimizer;

import com.codeberry.tadlib.nn.model.Model;

import static java.lang.Math.max;

public class DecayingLearningRate implements LearningRateSchedule {
    private final double minLr;
    private final int maxEpochsWithoutCostImprovement;
    private final double decayRate;

    private double lr;
    private int epochsWithoutImprovement;
    private int alreadyCheckedAtEpoch = -1;

    private double lowestCost = Double.MAX_VALUE;

    private DecayingLearningRate(double lr, double minLr, int maxEpochsWithoutCostImprovement, double decayRate) {
        this.lr = lr;
        this.minLr = minLr;
        this.maxEpochsWithoutCostImprovement = maxEpochsWithoutCostImprovement;
        this.decayRate = decayRate;
    }

    public static DecayingLearningRate decayingLearningRate(double lr, double minLr, int maxEpochsWithoutCostImprovement, double decayRate) {
        return new DecayingLearningRate(lr, minLr, maxEpochsWithoutCostImprovement, decayRate);
    }

    @Override
    public void beforeBatch(Model.IterationInfo iterationInfo) {
        if (alreadyCheckedAtEpoch != iterationInfo.epoch) {
            beforeEpoch(iterationInfo);
            alreadyCheckedAtEpoch = iterationInfo.epoch;
        }
    }

    private void beforeEpoch(Model.IterationInfo iterationInfo) {
        if (iterationInfo.hasPrevEpochTrainInfo()) {
            double trainingCost = iterationInfo.prevEpochTrainInfo.training.cost;
            if (lowestCost == Double.MAX_VALUE) {
                epochsWithoutImprovement = 0;
                lowestCost = trainingCost;
                System.out.println("New lowest cost: " + lowestCost);
            } else if (trainingCost < lowestCost) {
                epochsWithoutImprovement = 0;
                System.out.println("New lowest cost: " + lowestCost + " -> " + trainingCost);
                lowestCost = trainingCost;
            } else if (epochsWithoutImprovement < maxEpochsWithoutCostImprovement) {
                epochsWithoutImprovement++;
                System.out.println("No cost improvement: " + trainingCost +
                        "(" + lowestCost + "), decay in " + (maxEpochsWithoutCostImprovement - epochsWithoutImprovement) + " epochs");
            } else {
                epochsWithoutImprovement = 0;
                double newLr = max(minLr, lr * decayRate);
                System.out.println("Decay learning rate: " + lr + " -> " + newLr);
                lr = newLr;
            }
        }
    }

    @Override
    public double getLearningRate() {
        return lr;
    }
}
