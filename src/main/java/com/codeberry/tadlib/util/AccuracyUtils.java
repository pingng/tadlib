package com.codeberry.tadlib.util;

import com.codeberry.tadlib.array.NDIntArray;
import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.tensor.Tensor;

import java.util.Arrays;

public class AccuracyUtils {
    public static double softmaxAccuracy__(Tensor labels, Tensor prediction) {
        NDIntArray argMax = prediction.getVals().argmax(-1);

        System.out.println(StringUtils.toJson(argMax.toInts(), StringUtils.JsonPrintMode.COMPACT));
        System.out.println(StringUtils.toJson(labels.getVals().toDoubles(), StringUtils.JsonPrintMode.COMPACT));


        return 0;
    }

    public static double softmaxAccuracy(Tensor labels, Tensor prediction) {
        //softmaxAccuracy__(labels, prediction);

        Shape predictionShape = prediction.getShape();
        double[] predData = prediction.getInternalData();
        double[] lblData = labels.getInternalData();

        if (predictionShape.getDimCount() != 2) {
            throw new UnsupportedOperationException("expected 2 dims");
        }
        double acc = 0;
        int examples = predictionShape.at(0);
        for (int i = 0; i < examples; i++) {
            int predClass = maxIndex(predictionShape, predData, i);
            int lblOffset = labels.getShape().calcDataIndex(i, 0);
            int expectedClass = (int) lblData[lblOffset];
            if (predClass == expectedClass) {
                acc++;
            }
        }
        return acc/examples;
    }

    private static int maxIndex(Shape shape, double[] predData, int firstDimIndex) {
        int classDimLen = shape.at(-1);
        double max = Double.NEGATIVE_INFINITY;
        int maxIdx = -1;
        for (int i = 0; i < classDimLen; i++) {
            int predOffset = shape.calcDataIndex(firstDimIndex, i);
            double v = predData[predOffset];
            if (v > max) {
                max = v;
                maxIdx = i;
            }
        }
        return maxIdx;
    }
}
