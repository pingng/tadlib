package com.codeberry.tadlib.singlevalue;

/**
 * gradient link
 */
public class GradLink {
    public final Value value;
    public final GradFunc gradFunc;

    public GradLink(Value value, GradFunc gradFunc) {
        this.value = value;
        this.gradFunc = gradFunc;
    }

    public static GradLink dependency(Value value, GradFunc gradFunc) {
        return new GradLink(value, gradFunc);
    }
}
