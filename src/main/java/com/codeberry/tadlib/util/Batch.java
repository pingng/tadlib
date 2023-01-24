package com.codeberry.tadlib.util;

import com.codeberry.tadlib.tensor.Tensor;

public class Batch {
    public final Tensor input;
    public final Tensor output;

    public Batch(Tensor input, Tensor output) {
        this.input = input;
        this.output = output;
    }

    public int getBatchSize() {
        return input.shape().at(0);
    }
}
