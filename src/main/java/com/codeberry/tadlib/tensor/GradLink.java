package com.codeberry.tadlib.tensor;

public class GradLink {
    final Tensor tensor;
    final GradFunc gradFunc;

    GradLink(Tensor tensor, GradFunc gradFunc) {
        this.tensor = tensor;
        this.gradFunc = gradFunc;
    }

    static GradLink grad(Tensor parent, GradFunc gradFunc) {
        return new GradLink(parent, gradFunc);
    }
}
