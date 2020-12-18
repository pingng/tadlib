package com.codeberry.tadlib.util;

import com.codeberry.tadlib.array.TMutableArray;
import com.codeberry.tadlib.tensor.Tensor;

import static com.codeberry.tadlib.array.Shape.*;

public abstract class TrainingDataUtils {
    public static Tensor toOneHot(Tensor yTrain, int outputUnits) {
        int examples = yTrain.getShape().at(0);
        TMutableArray out = new TMutableArray(new double[examples * outputUnits], shape(examples, outputUnits));
        int[] indices = out.shape.newIndexArray();
        for (int i = 0; i < examples; i++) {
            indices[0] = i;
            indices[1] = (int) yTrain.dataAt(i, 0);
            out.setAt(indices, 1.0);
        }
        return new Tensor(out.migrateToImmutable(), Tensor.GradientMode.NONE);
    }
}
