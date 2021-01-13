package com.codeberry.tadlib.nn.loss;

import com.codeberry.tadlib.array.NDArray;
import com.codeberry.tadlib.tensor.Ops;

public class SoftmaxCrossEntropyLoss {
    public static double sumSoftmaxCrossEntropy(NDArray predicted, int[] indices, NDArray target, int dim) {
        double[] predData = predicted.getInternalData();
        double[] targetData = target.getInternalData();

        return sumSoftmaxCrossEntropy(predicted, predData, indices, target, targetData, dim);
    }

    private static double sumSoftmaxCrossEntropy(NDArray predicted, double[] predData, int[] indices, NDArray target, double[] targetData, int dim) {
        int len = predicted.getShape().at(dim);
        if (indices.length - dim == 1) {
            double sum = 0;
            for (int i = 0; i < len; i++) {
                indices[dim] = i;
                int tgtOffset = target.getShape().calcDataIndex(indices);
                double tgt = targetData[tgtOffset];
                int predOffset = predicted.getShape().calcDataIndex(indices);
                double pred = predData[predOffset];
                if (pred < Ops.EPSILON) {
                    pred = Ops.EPSILON;
                } else if (pred > 1.0 - Ops.EPSILON) {
                    pred = 1.0 - Ops.EPSILON;
                }
                sum += -tgt * Math.log(pred);
            }
            return sum;
        } else {
            double sum = 0;
            for (int i = 0; i < len; i++) {
                indices[dim] = i;
                sum += sumSoftmaxCrossEntropy(predicted, indices, target, dim + 1);
            }
            return sum;
        }
    }
}
