package com.codeberry.tadlib.singlevalue;

public class ParentLink {
    public final Value value;
    public final GradFunc gradFunc;

    public ParentLink(Value value, GradFunc gradFunc) {
        this.value = value;
        this.gradFunc = gradFunc;
    }

    public static ParentLink dependency(Value value, GradFunc gradFunc) {
        return new ParentLink(value, gradFunc);
    }
}
