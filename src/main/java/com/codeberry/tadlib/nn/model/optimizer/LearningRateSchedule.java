package com.codeberry.tadlib.nn.model.optimizer;

import com.codeberry.tadlib.nn.model.Model;

public interface LearningRateSchedule {
    default void beforeBatch(Model.IterationInfo iterationInfo) {
        // do nothing
    }

    double getLearningRate();
}
