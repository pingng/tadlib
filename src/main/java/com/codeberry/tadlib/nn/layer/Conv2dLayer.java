package com.codeberry.tadlib.nn.layer;

import com.codeberry.tadlib.nn.loss.L2Loss;
import com.codeberry.tadlib.provider.java.Shape;
import com.codeberry.tadlib.tensor.Ops;
import com.codeberry.tadlib.tensor.Tensor;

import java.util.Random;

import static com.codeberry.tadlib.nn.Model.IterationInfo;
import static com.codeberry.tadlib.nn.layer.Layer.ForwardResult.result;
import static com.codeberry.tadlib.provider.ProviderStore.shape;
import static com.codeberry.tadlib.tensor.Ops.RunMode;
import static com.codeberry.tadlib.tensor.Ops.add;
import static com.codeberry.tadlib.tensor.Tensor.TensorFactories.randomWeight;
import static com.codeberry.tadlib.tensor.Tensor.TensorFactories.zeros;

public class Conv2dLayer implements Layer {
    private final Tensor w;
    private final Tensor b;
    private final double l2Lambda;

    private final Shape outputShape;

    public Conv2dLayer(Random rnd, Shape inputShape, Builder params) {
        w = randomWeight(rnd, shape(params.kernelSize, params.kernelSize, inputShape.at(-1), params.filters));
        b = (params.biasParam == BiasParam.USE_BIAS ? zeros(shape(params.filters)) : null);

        l2Lambda = params.l2Lambda;
        outputShape = inputShape.withDimAt(-1, params.filters);
    }

    @Override
    public ForwardResult forward(Random rnd, Tensor inputs, RunMode runMode, IterationInfo iterationInfo) {
        Tensor hW = Ops.conv2d(inputs, w);
        if (b != null) {
            return result(add(hW, b));
        }
        return result(hW);
    }

    @Override
    public Tensor getAdditionalCost() {
        if (l2Lambda > 0) {
            return L2Loss.l2LossOf(l2Lambda, w);
        }
        return null;
    }

    @Override
    public Shape getOutputShape() {
        return outputShape;
    }

    @Override
    public Tensor[] getTrainableParamsNullable() {
        return new Tensor[]{w, b};
    }

    public static class Builder implements LayerBuilder {
        int kernelSize;
        int filters;
        double l2Lambda;
        BiasParam biasParam = BiasParam.USE_BIAS;

        @Override
        public Layer build(Random rnd, Shape inputShape) {
            return new Conv2dLayer(rnd, inputShape, this);
        }

        public static Builder conv2d() {
            return new Builder();
        }

        public Builder kernelSize(int kernelSize) {
            this.kernelSize = kernelSize;
            return this;
        }

        public Builder filters(int filters) {
            this.filters = filters;
            return this;
        }

        public Builder l2Lambda(double l2Lambda) {
            this.l2Lambda = l2Lambda;
            return this;
        }

        public Builder biasParam(BiasParam biasParam) {
            this.biasParam = biasParam;
            return this;
        }

    }

    public enum BiasParam {
        USE_BIAS, NO_BIAS
    }
}
