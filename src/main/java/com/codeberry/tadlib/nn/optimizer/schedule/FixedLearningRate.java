package com.codeberry.tadlib.nn.optimizer.schedule;

public class FixedLearningRate implements LearningRateSchedule {
    private final double lr;

    private FixedLearningRate(double lr) {
        this.lr = lr;
    }

    public static FixedLearningRate fixedLearningRate(double lr) {
        return new FixedLearningRate(lr);
    }

    @Override
    public double getLearningRate() {
        return lr;
    }
}
