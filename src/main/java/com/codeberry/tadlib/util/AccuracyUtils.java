package com.codeberry.tadlib.util;

import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.tensor.Tensor;

public class AccuracyUtils {
    public static double softmaxAccuracy(Tensor labels, Tensor prediction) {
        Shape predictionShape = prediction.getShape();

        if (predictionShape.dimCount != 2) {
            throw new UnsupportedOperationException("expected 2 dims");
        }
        double acc = 0;
        int examples = predictionShape.at(0);
        for (int i = 0; i < examples; i++) {
            int predClass = maxIndex(prediction, i);
            int expectedClass = (int) labels.dataAt(i, 0);
            if (predClass == expectedClass) {
                acc++;
            }
        }
        return acc/examples;
    }

    private static int maxIndex(Tensor pred, int firstDim) {
        int classDimLen = pred.getShape().at(-1);
        double max = Double.NEGATIVE_INFINITY;
        int maxIdx = -1;
        for (int i = 0; i < classDimLen; i++) {
            double v = pred.dataAt(firstDim, i);
            if (v > max) {
                max = v;
                maxIdx = i;
            }
        }
        return maxIdx;
    }
}
