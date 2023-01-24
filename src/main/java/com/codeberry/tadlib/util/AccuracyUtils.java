package com.codeberry.tadlib.util;

import com.codeberry.tadlib.provider.java.JavaIntArray;
import com.codeberry.tadlib.provider.java.NDArray;
import com.codeberry.tadlib.tensor.Tensor;

import static com.codeberry.tadlib.array.Comparison.equalsWithDelta;

public class AccuracyUtils {
    public static double softmaxAccuracy(Tensor labels, Tensor prediction) {
        // TODO: test shapes
        JavaIntArray predictedClasses = prediction.val().argmax(-1);
        NDArray labelClasses = labels.val().reshape(-1);
        int examples = labelClasses.shape.at(0);

        NDArray equals = labelClasses.compare(predictedClasses, equalsWithDelta(1.0E-20), 1.0, 0.0);
        NDArray sum = equals.sum();
        NDArray acc = sum.div(examples);
        return (Double) acc.toDoubles();
    }
}
