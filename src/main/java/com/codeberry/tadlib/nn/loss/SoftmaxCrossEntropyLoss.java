package com.codeberry.tadlib.nn.loss;

import com.codeberry.tadlib.array.TArray;
import com.codeberry.tadlib.tensor.Ops;

public class SoftmaxCrossEntropyLoss {
    public static double sumSoftmaxCrossEntropy(TArray predicted, int[] indices, TArray target, int dim) {
        int len = predicted.shape.at(dim);
        if (indices.length - dim == 1) {
            double sum = 0;
            for (int i = 0; i < len; i++) {
                indices[dim] = i;
                double tgt = target.dataAt(indices);
                double pred = predicted.dataAt(indices);
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
