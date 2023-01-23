package com.codeberry.tadlib.nn.loss;

import com.codeberry.tadlib.provider.java.NDArray;
import com.codeberry.tadlib.tensor.Ops;

public class SoftmaxCrossEntropyLoss {
    /**
     * Calculates SUM(-target * log(prediction))
     */
    public static NDArray sumSoftmaxCrossEntropy(NDArray predicted, NDArray target) {
        NDArray clippedPrediction = predicted.clip(Ops.EPSILON, 1.0 - Ops.EPSILON);
        NDArray logPrediction = clippedPrediction.log();
        NDArray prod = target.mul(logPrediction);
        NDArray sum = prod.sum();

        return sum.negate();
    }
}
