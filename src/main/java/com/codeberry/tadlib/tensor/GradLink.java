package com.codeberry.tadlib.tensor;

public class GradLink {
    final Tensor parent;
    final GradFunc gradFunc;

    GradLink(Tensor parent, GradFunc gradFunc) {
        this.parent = parent;
        this.gradFunc = gradFunc;
    }

    static GradLink grad(Tensor parent, GradFunc gradFunc) {
        return new GradLink(parent, gradFunc);
    }
}
