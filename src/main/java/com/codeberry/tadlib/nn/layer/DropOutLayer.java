package com.codeberry.tadlib.nn.layer;

import com.codeberry.tadlib.nn.Model.IterationInfo;
import com.codeberry.tadlib.provider.java.Shape;
import com.codeberry.tadlib.tensor.Ops;
import com.codeberry.tadlib.tensor.Tensor;

import java.util.Random;

import static com.codeberry.tadlib.nn.layer.Layer.ForwardResult.result;
import static com.codeberry.tadlib.tensor.Ops.RunMode;

public class DropOutLayer implements Layer {
    private final double dropoutKeep;
    private final Shape outputShape;

    public DropOutLayer(Shape inputShape, Builder params) {
        this.dropoutKeep = params.dropoutKeep;

        outputShape = inputShape;
    }

    @Override
    public ForwardResult forward(Random rnd, Tensor inputs, RunMode runMode, IterationInfo iterationInfo) {
        return result(Ops.dropout(inputs, rnd, dropoutKeep, runMode));
    }

    @Override
    public Shape getOutputShape() {
        return outputShape;
    }

    public static class Builder implements LayerBuilder {

        @Override
        public Layer build(Random rnd, Shape inputShape) {
            return new DropOutLayer(inputShape, this);
        }

        double dropoutKeep;

        public static Builder dropout() {
            return new Builder();
        }

        public Builder dropoutKeep(double dropoutKeep) {
            this.dropoutKeep = dropoutKeep;
            return this;
        }
    }
}
