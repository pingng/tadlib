package com.codeberry.tadlib.nn.optimizer.schedule;

import com.codeberry.tadlib.nn.Model;

public interface LearningRateSchedule {
    default void beforeBatch(Model.IterationInfo iterationInfo) {
        // do nothing
    }

    double getLearningRate();
}
