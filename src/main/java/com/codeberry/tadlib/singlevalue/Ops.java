package com.codeberry.tadlib.singlevalue;

import static com.codeberry.tadlib.singlevalue.ParentLink.dependency;
import static java.util.Arrays.asList;
import static java.util.Collections.singletonList;

public class Ops {
    public static Value add(Value a, Value b) {
        double v = a.v + b.v;

        GradFunc gradFn = grad -> grad;

        return new Value(v,
                asList(dependency(a, gradFn), dependency(b, gradFn)));
    }

    public static Value sub(Value a, Value b) {
        double v = a.v - b.v;

        GradFunc gradFn_a = grad -> grad;
        GradFunc gradFn_b = grad -> -grad;

        return new Value(v,
                asList(dependency(a, gradFn_a), dependency(b, gradFn_b)));
    }

    public static Value mul(Value a, Value b) {
        double v = a.v * b.v;

        GradFunc gradFn_a = grad -> grad * b.v;
        GradFunc gradFn_b = grad -> grad * a.v;

        return new Value(v,
                asList(dependency(a, gradFn_a), dependency(b, gradFn_b)));
    }

    public static Value sqr(Value a) {
        double v = a.v * a.v;

        GradFunc gradFn = grad -> 2 * a.v * grad;

        return new Value(v, singletonList(dependency(a, gradFn)));
    }

    public static Value sin(Value a) {
        double v = Math.sin(a.v);

        GradFunc gradFn = grad -> Math.cos(a.v) * grad;

        return new Value(v, singletonList(dependency(a, gradFn)));
    }

    public static Value neg(Value a) {
        double v = -a.v;

        GradFunc gradFn = grad -> -grad;

        return new Value(v, singletonList(dependency(a, gradFn)));
    }

    public static Value tanh(Value a) {
        double v = Math.tanh(a.v);

        GradFunc gradFn = grad -> (1 - v * v) * grad;

        return new Value(v, singletonList(dependency(a, gradFn)));
    }

    public static Value pow(Value a, Value b) {
        double v = Math.pow(a.v, b.v);

        GradFunc gradFn_a = grad -> b.v * Math.pow(a.v, b.v - 1.0) * grad;
        GradFunc gradFn_b = grad -> v * Math.log(a.v) * grad;

        return new Value(v,
                asList(dependency(a, gradFn_a),
                        dependency(b, gradFn_b)));
    }
}
