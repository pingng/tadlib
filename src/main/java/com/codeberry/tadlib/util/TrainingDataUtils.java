package com.codeberry.tadlib.util;

import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.provider.java.TMutableArray;
import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.tensor.Tensor;

import static com.codeberry.tadlib.provider.ProviderStore.shape;

public abstract class TrainingDataUtils {
    public static Tensor toOneHot(Tensor yTrain, int outputUnits) {
        Shape shape = yTrain.getShape();
        if (shape.getDimCount() != 2) {
            throw new IllegalArgumentException("Expected 2 dimensions (batch, labelValue): actualShape=" + shape);
        }
        if (shape.at(-1) != 1) {
            throw new IllegalArgumentException("Expected last dimension to be of length 1 (the label value): actualLen=" + shape.at(-1));
        }

        int examples = shape.at(0);
        TMutableArray out = new TMutableArray(new double[examples * outputUnits], shape(examples, outputUnits));
        int[] indices = out.shape.newIndexArray();
        for (int i = 0; i < examples; i++) {
            indices[0] = i;
            indices[1] = (int) yTrain.dataAt(i, 0);
            out.setAt(indices, 1.0);
        }
        return new Tensor(ProviderStore.array(out.getData(), shape(examples, outputUnits)), Tensor.GradientMode.NONE);
    }
}
