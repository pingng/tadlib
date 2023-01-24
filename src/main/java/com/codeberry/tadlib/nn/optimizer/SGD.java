package com.codeberry.tadlib.nn.optimizer;

import com.codeberry.tadlib.nn.optimizer.schedule.LearningRateSchedule;
import com.codeberry.tadlib.provider.java.NDArray;
import com.codeberry.tadlib.tensor.Tensor;

import java.util.List;
import java.util.function.BiFunction;

public record SGD(LearningRateSchedule learningRateSchedule) implements Optimizer {

    @Override
    public void optimize(List<Tensor> params) {
        double lr = learningRateSchedule.getLearningRate();

        BiFunction<NDArray, NDArray, NDArray> each = (values, gradient) ->
            values.sub(gradient.mul(lr));

        for (Tensor p : params)
            p.update(each);
    }

}
