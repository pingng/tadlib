package com.codeberry.tadlib.nn.model.optimizer;

import com.codeberry.tadlib.tensor.Tensor;

import java.util.List;

public class SGD implements Optimizer {
    private final double learningRate;

    public SGD(double learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public void optimize(List<Tensor> params) {
        for (Tensor p : params) {
            p.update((values, gradient) -> values.sub(gradient.mul(learningRate)));
        }
    }
}
