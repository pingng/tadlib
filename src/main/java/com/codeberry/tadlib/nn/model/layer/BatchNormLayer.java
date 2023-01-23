package com.codeberry.tadlib.nn.model.layer;

import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.memorymanagement.DisposalRegister;
import com.codeberry.tadlib.nn.model.Model;
import com.codeberry.tadlib.tensor.Tensor;

import java.util.List;
import java.util.Random;

import static com.codeberry.tadlib.nn.model.layer.Layer.ForwardResult.result;
import static com.codeberry.tadlib.provider.ProviderStore.shape;
import static com.codeberry.tadlib.tensor.Ops.*;
import static com.codeberry.tadlib.tensor.OpsExtended.*;
import static com.codeberry.tadlib.tensor.Tensor.TensorFactories.*;

public class BatchNormLayer implements Layer {
    private final double runningAverageMomentum;
    private final Tensor beta;
    private final Tensor gamma;

    private final BatchNormRunningAverages runningAverages = new BatchNormRunningAverages();

    private final Shape outputShape;

    public BatchNormLayer(Random rnd, Shape inputShape, Builder params) {
        int paramsLen = guessParamLength(inputShape);

        this.beta = zeros(shape(paramsLen));
        this.gamma = ones(shape(paramsLen));
        this.runningAverageMomentum = params.runningAverageMomentum;

        outputShape = inputShape;
    }

    @Override
    public ForwardResult forward(Random rnd, Tensor inputs, RunMode runMode, Model.IterationInfo iterationInfo) {
        BatchNormResult result = batchNorm(inputs, beta, gamma, runningAverages, runMode);

        if (runMode == RunMode.TRAINING) {
            return result(result.output,
                    () -> this.runningAverages.updateWith(result, runningAverageMomentum));
        }
        return result(result.output);
    }

    @Override
    public String getTrainingSummary() {
        return "RunningAvg: " + runningAverages;
    }

    @Override
    public Shape getOutputShape() {
        return outputShape;
    }

    @Override
    public Tensor[] getTrainableParams() {
        return new Tensor[]{beta, gamma};
    }

    @Override
    public List<? extends DisposalRegister.Disposable> getKeepInMemoryDisposables() {
        return runningAverages.getKeepInMemoryDisposables();
    }

    public static class Builder implements LayerBuilder {

        double runningAverageMomentum = 0.99;

        @Override
        public Layer build(Random rnd, Shape inputShape) {
            return new BatchNormLayer(rnd, inputShape, this);
        }

        public Builder runningAverageMomentum(double runningAverageMomentum) {
            this.runningAverageMomentum = runningAverageMomentum;
            return this;
        }

        public static Builder batchNorm() {
            return new Builder();
        }

    }
}
