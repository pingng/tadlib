package com.codeberry.tadlib.nn.loss;

import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.tensor.Ops;
import com.codeberry.tadlib.tensor.Tensor;

import java.util.Arrays;

import static com.codeberry.tadlib.tensor.Ops.*;
import static com.codeberry.tadlib.tensor.Tensor.constant;

public class L2Loss {
    public static Tensor l2LossOf(Shape inputShape, double l2Lambda, Tensor... tensors) {
        int actualBatchSize = inputShape.at(0);

        Tensor l2Scale = constant(l2Lambda / (2 * actualBatchSize));

        Tensor[] l2Costs = Arrays.stream(tensors)
                .map(t -> mul(sum(sqr(t)), l2Scale))
                .toArray(Tensor[]::new);

        return Ops.add(l2Costs);
    }
}
