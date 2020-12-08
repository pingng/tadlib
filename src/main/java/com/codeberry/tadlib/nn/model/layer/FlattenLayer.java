package com.codeberry.tadlib.nn.model.layer;

import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.tensor.Ops;
import com.codeberry.tadlib.tensor.Tensor;

import java.util.Random;

import static com.codeberry.tadlib.array.Shape.shape;
import static com.codeberry.tadlib.nn.model.layer.Layer.ForwardResult.result;
import static com.codeberry.tadlib.tensor.Ops.RunMode;

public class FlattenLayer implements Layer {

    private final Shape outputShape;

    public FlattenLayer(Random rnd, Shape inputShape, Builder params) {
        int exampleSize = Ops.calcFlattenExampleSize(inputShape);

        outputShape = shape(-1, exampleSize);
    }

    @Override
    public ForwardResult forward(Random rnd, Tensor inputs, RunMode runMode) {
        return result(Ops.flatten(inputs));
    }

    @Override
    public Shape getOutputShape() {
        return outputShape;
    }

    public static class Builder implements LayerBuilder {

        @Override
        public Layer build(Random rnd, Shape inputShape) {
            return new FlattenLayer(rnd, inputShape, this);
        }

        public static Builder flatten() {
            return new Builder();
        }
    }

    public enum BiasParam {
        USE_BIAS, NO_BIAS
    }
}
