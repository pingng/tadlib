package com.codeberry.tadlib.nn.model;

import com.codeberry.tadlib.tensor.Tensor;

import java.util.List;

public interface Optimizer {
    void optimize(List<Tensor> params);
}
