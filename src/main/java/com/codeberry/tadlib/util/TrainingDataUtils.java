package com.codeberry.tadlib.util;

import com.codeberry.tadlib.array.TArray;
import com.codeberry.tadlib.tensor.Tensor;

public abstract class TrainingDataUtils {
    public static Tensor toOneHot(Tensor yTrain, int outputUnits) {
        int examples = yTrain.getShape().at(0);
        TArray out = new TArray(new double[examples][outputUnits]);
        int[] indices = out.shape.newIndexArray();
        for (int i = 0; i < examples; i++) {
            indices[0] = i;
            indices[1] = (int) yTrain.dataAt(i, 0);
            out.setAt(indices, 1.0);
        }
        return new Tensor(out, Tensor.GradientMode.NONE);
    }
}
