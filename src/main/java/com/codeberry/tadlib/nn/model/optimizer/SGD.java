package com.codeberry.tadlib.nn.model.optimizer;

import com.codeberry.tadlib.tensor.Tensor;

import java.util.List;

public class SGD implements Optimizer {
    private final LearningRateSchedule learningRateSchedule;

    public SGD(LearningRateSchedule learningRateSchedule) {
        this.learningRateSchedule = learningRateSchedule;
    }

    @Override
    public LearningRateSchedule getLearningRateSchedule() {
        return learningRateSchedule;
    }

    @Override
    public void optimize(List<Tensor> params) {
        for (Tensor p : params) {
            p.update((values, gradient) -> values.sub(gradient.mul(learningRateSchedule.getLearningRate())));
        }
    }
}
