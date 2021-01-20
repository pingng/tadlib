package com.codeberry.tadlib.util;

import com.codeberry.tadlib.array.NDArray;
import com.codeberry.tadlib.array.NDIntArray;
import com.codeberry.tadlib.tensor.Tensor;

import static com.codeberry.tadlib.array.Comparison.equalsWithDelta;

public class AccuracyUtils {
    public static double softmaxAccuracy(Tensor labels, Tensor prediction) {
        // TODO: test shapes
        NDIntArray predictedClasses = prediction.getVals().argmax(-1);
        NDArray labelClasses = labels.getVals().reshape(-1);
        int examples = labelClasses.getShape().at(0);

        NDArray equals = labelClasses.compare(predictedClasses, equalsWithDelta(1E-20), 1., 0.);
        NDArray sum = equals.sum();
        NDArray acc = sum.div(examples);
        return (Double) acc.toDoubles();
    }
}
