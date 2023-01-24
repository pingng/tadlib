package com.codeberry.tadlib.nn.layer;

import com.codeberry.tadlib.provider.java.Shape;
import com.codeberry.tadlib.tensor.Ops;
import com.codeberry.tadlib.tensor.Tensor;

import java.util.Random;

import static com.codeberry.tadlib.nn.Model.IterationInfo;
import static com.codeberry.tadlib.nn.layer.Layer.ForwardResult.result;
import static com.codeberry.tadlib.provider.ProviderStore.shape;
import static com.codeberry.tadlib.tensor.Ops.RunMode;

public class FlattenLayer implements Layer {

    private final Shape outputShape;

    public FlattenLayer(Shape inputShape) {
        int exampleSize = Ops.calcFlattenExampleSize(inputShape);

        outputShape = shape(-1, exampleSize);
    }

    @Override
    public ForwardResult forward(Random rnd, Tensor inputs, RunMode runMode, IterationInfo iterationInfo) {
        return result(Ops.flatten(inputs));
    }

    @Override
    public Shape getOutputShape() {
        return outputShape;
    }

    public static class Builder implements LayerBuilder {

        @Override
        public Layer build(Random rnd, Shape inputShape) {
            return new FlattenLayer(inputShape);
        }

        public static Builder flatten() {
            return new Builder();
        }
    }

    public enum BiasParam {
        USE_BIAS, NO_BIAS
    }
}
