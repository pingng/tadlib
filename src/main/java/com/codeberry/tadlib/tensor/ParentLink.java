package com.codeberry.tadlib.tensor;

public class ParentLink {
    final Tensor parent;
    final GradFunc gradFunc;

    ParentLink(Tensor parent, GradFunc gradFunc) {
        this.parent = parent;
        this.gradFunc = gradFunc;
    }

    static ParentLink parent(Tensor parent, GradFunc gradFunc) {
        return new ParentLink(parent, gradFunc);
    }
}
