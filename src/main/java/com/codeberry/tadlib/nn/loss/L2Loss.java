package com.codeberry.tadlib.nn.loss;

import com.codeberry.tadlib.tensor.Ops;
import com.codeberry.tadlib.tensor.Tensor;

import java.util.Arrays;

import static com.codeberry.tadlib.tensor.Ops.*;
import static com.codeberry.tadlib.tensor.Tensor.constant;

public class L2Loss {
    public static Tensor l2LossOf(double l2Lambda, Tensor... tensors) {
        Tensor l2Scale = constant(l2Lambda / 2.0);

        Tensor[] l2Costs = Arrays.stream(tensors)
                .map(t -> mul(sum(sqr(t)), l2Scale))
                .toArray(Tensor[]::new);

        return Ops.add(l2Costs);
    }
}
