package com.codeberry.tadlib.nn.optimizer.schedule;

public class FixedLearningRate implements LearningRateSchedule {
    private final double lr;

    public FixedLearningRate(double lr) {
        this.lr = lr;
    }

    @Override
    public double getLearningRate() {
        return lr;
    }
}
