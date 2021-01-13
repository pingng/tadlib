package com.codeberry.tadlib.nn.model.layer;

import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.array.util.DimensionUtils;
import com.codeberry.tadlib.tensor.Tensor;

import java.util.Random;

import static com.codeberry.tadlib.nn.model.layer.Layer.ForwardResult.*;
import static com.codeberry.tadlib.tensor.Ops.*;

public class MaxPool2dLayer implements Layer {
    private final int size;
    private final Shape outputShape;

    public MaxPool2dLayer(Shape inputShape, Builder params) {
        this.size = params.size;

        outputShape = DimensionUtils.getMaxPool2dResultShape(inputShape, size);
    }

    @Override
    public ForwardResult forward(Random rnd, Tensor inputs, RunMode runMode) {
        return result(maxpool2d(inputs, size));
    }

    @Override
    public Shape getOutputShape() {
        return outputShape;
    }

    public static class Builder implements LayerBuilder {

        int size;

        @Override
        public Layer build(Random rnd, Shape inputShape) {
            return new MaxPool2dLayer(inputShape, this);
        }

        public static Builder maxPool2d() {
            return new Builder();
        }

        public Builder size(int size) {
            this.size = size;
            return this;
        }
    }
}
