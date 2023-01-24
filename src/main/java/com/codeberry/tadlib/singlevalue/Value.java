package com.codeberry.tadlib.singlevalue;

import java.util.List;

import static java.util.Collections.emptyList;

public class Value {
    public final double v;
    private final GradLink[] dependencies;

    public double grad;
    private final boolean hasGradient;

    public Value(double v) {
        this(v, emptyList(), true);
    }

    public Value(double v, List<GradLink> dependencies) {
        this(v, dependencies, true);
    }

    public Value(double v, List<GradLink> dependencies, boolean hasGradient) {
        this.v = v;
        this.dependencies = dependencies.toArray(GradLink[]::new);
        this.hasGradient = hasGradient;
    }

    public static Value value(double v) {
        return new Value(v);
    }

    public static Value constant(double v) {
        return new Value(v, emptyList(), false);
    }

    public void backward() {
        backward(1.0);
    }

    public void backward(double grad) {
        if (hasGradient) {
            this.grad += grad;
            for (var d : dependencies)
                d.value.backward(d.gradFunc.calcGradient(grad));
        }
    }

    @Override
    public String toString() {
        return String.valueOf(v);
    }
}
