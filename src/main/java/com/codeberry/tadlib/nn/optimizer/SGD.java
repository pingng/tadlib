package com.codeberry.tadlib.nn.optimizer;

import com.codeberry.tadlib.nn.optimizer.schedule.LearningRateSchedule;
import com.codeberry.tadlib.provider.java.NDArray;
import com.codeberry.tadlib.tensor.Tensor;

import java.util.List;
import java.util.function.BiFunction;

public record SGD(LearningRateSchedule learningRateSchedule) implements Optimizer {

    @Override
    public void optimize(List<Tensor> params) {
        BiFunction<NDArray, NDArray, NDArray> each = this::apply;
        for (Tensor p : params)
            p.update(each);
    }

    private NDArray apply(NDArray values, NDArray gradient) {
        return values.sub(gradient.mul(learningRateSchedule.getLearningRate()));
    }
}
