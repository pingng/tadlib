package com.codeberry.tadlib.example.mnist;

import com.codeberry.tadlib.array.TArray;
import com.codeberry.tadlib.example.TrainingData;
import com.codeberry.tadlib.tensor.Tensor;
import com.codeberry.tadlib.util.ReflectionUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static com.codeberry.tadlib.array.Shape.shape;
import static com.codeberry.tadlib.array.TArray.value;
import static com.codeberry.tadlib.example.mnist.MNISTLoader.*;
import static com.codeberry.tadlib.nn.loss.L2Loss.l2LossOf;
import static com.codeberry.tadlib.tensor.Ops.*;
import static com.codeberry.tadlib.tensor.OpsExtended.*;
import static com.codeberry.tadlib.tensor.Tensor.TensorFactories.*;
import static com.codeberry.tadlib.tensor.Tensor.constant;
import static com.codeberry.tadlib.util.AccuracyUtils.*;

public class MNISTConvModel {
    private final Config cfg;

    private final Tensor w;
    private final Tensor b;
    private final Tensor sec_w;
    private final Tensor sec_b;
    private final Tensor fullW;
    private final Tensor fullB;
    private final Tensor finalW;
    private final Tensor finalB;
    private final Tensor sec_bn_beta;
    private final Tensor sec_bn_gamma;
    private final Tensor full_bn_beta;
    private final Tensor full_bn_gamma;

    public BatchNormRunningAverages sec_bnAverages = new BatchNormRunningAverages();
    public BatchNormRunningAverages full_bnAverages = new BatchNormRunningAverages();

    private final List<Runnable> additionalUpdatesOnWeightUpdate = new ArrayList<>();

    public MNISTConvModel(Config cfg) {
        this.cfg = cfg;

        int imageSize = 28;
        int out = 10;

        Random r = new Random(cfg.weightInitRandomSeed);
        this.w = randomWeight(r, shape(3, 3, 1, cfg.firstConvChannels));
        this.b = zeros(shape(cfg.firstConvChannels));

        this.sec_w = randomWeight(r, shape(3, 3, cfg.firstConvChannels, cfg.secondConvChannels));

        if (cfg.useBatchNormalization) {
            this.sec_b = null;
            this.fullB = null;

            this.sec_bn_beta = zeros(shape(cfg.secondConvChannels));
            this.sec_bn_gamma = ones(shape(cfg.secondConvChannels));
            this.full_bn_beta = zeros(shape(1));
            this.full_bn_gamma = ones(shape(1));
        } else {
            this.sec_b = zeros(shape(cfg.secondConvChannels));
            this.fullB = zeros(shape(cfg.fullyConnectedSize));

            this.sec_bn_beta = null;
            this.sec_bn_gamma = null;
            this.full_bn_beta = null;
            this.full_bn_gamma = null;
        }

        int hiddenSize = imageSize / 2 / 2;
        this.fullW = randomWeight(r,
                shape(hiddenSize * hiddenSize * cfg.secondConvChannels,
                        cfg.fullyConnectedSize));

        this.finalW = randomWeight(r, shape(cfg.fullyConnectedSize, out));
        this.finalB = zeros(shape(out));
    }

    private MNISTConvModel(MNISTConvModel src) {
        // init with dummy tensors for weights
        this(src.cfg);

        // overwrite weights using reflection
        ReflectionUtils.copyFieldOfClass(Tensor.class,
                src, this,
                Tensor::copy);
    }

    public void trainSingleIteration(Random rnd, TrainingData batchData, double lr, TrainStats stats) {
        calcGradient(rnd, batchData.xTrain, batchData.yTrain, stats);
        updateWeights(lr);
    }

    public void calcGradient(Random dropRnd,
                             Tensor xTrain, Tensor yTrain) {
        calcGradient(dropRnd, xTrain, yTrain, null);
    }

    public void calcGradient(Random rnd,
                             Tensor xTrain, Tensor yTrain,
                             TrainStats stats) {
        resetGradients();

        Tensor cost = calcCost(rnd, xTrain, yTrain, stats);
        cost.backward(value(1.0));
    }

    public Tensor calcCost(Random rnd,
                           Tensor xTrain, Tensor yTrain,
                           TrainStats stats) {
        int actualBatchSize = xTrain.getShape().at(0);

        Tensor y = forward(rnd, xTrain, RunMode.TRAINING);

        Tensor totalSoftmaxCost = sumSoftmaxCrossEntropy(toOneHot(yTrain), y);
        Tensor avgSoftmaxCost = div(totalSoftmaxCost, constant(actualBatchSize));

        Tensor l2Loss = cfg.l2Lambda <= 0 ? Tensor.ZERO :
                l2LossOf(xTrain.getShape(), cfg.l2Lambda,
                        w, sec_w, fullW, finalW);

        if (stats != null) {
            stats.accumulate((double) avgSoftmaxCost.toDoubles(),
                    (double) l2Loss.toDoubles(),
                    softmaxAccuracy(yTrain, y));
        }

        return add(avgSoftmaxCost, l2Loss);
    }

    public List<Tensor> getParams() {
        return ReflectionUtils.getFieldValues(Tensor.class, this);
    }

    public void updateWeights(double lr) {
        List<Tensor> params = getParams();

        for (Tensor p : params) {
            p.update((values, gradient) -> values.sub(gradient.mul(lr)));
        }
        for (Runnable update : additionalUpdatesOnWeightUpdate) {
            update.run();
        }
        additionalUpdatesOnWeightUpdate.clear();
    }

    public Tensor predict(Tensor x_train) {
        return forward(null, x_train, RunMode.INFERENCE);
    }

    private Tensor forward(Random rnd, Tensor inputs, RunMode runMode) {
        additionalUpdatesOnWeightUpdate.clear();

        Tensor firstLayerOut = firstConvLayer(inputs);
        Tensor secondLayerOut = secondConvLayer(runMode, firstLayerOut);
        Tensor fullyLayerOut = fullyConnectedLayer(rnd, secondLayerOut, runMode);
        return finalOutputLayer(fullyLayerOut);
    }

    private Tensor firstConvLayer(Tensor inputs) {
        Tensor h_w = conv2d(inputs, w);
        Tensor h = add(h_w, b);
        Tensor maxed = maxpool2d(h, 2);
        return relu(maxed);
    }

    private Tensor secondConvLayer(RunMode runMode, Tensor inputs) {
        Tensor sec_h_w = conv2d(inputs, sec_w);
        Tensor secOut = cfg.useBatchNormalization ?
                sec_h_w : add(sec_h_w, sec_b);

        Tensor sec_maxed = maxpool2d(secOut, 2);
        Tensor secRelu = relu(sec_maxed);

        if (cfg.useBatchNormalization) {
            BatchNormResult secBnResult = batchNorm(secRelu, sec_bn_beta, sec_bn_gamma, sec_bnAverages, runMode);
            additionalUpdatesOnWeightUpdate.add(() ->
                    this.sec_bnAverages = this.sec_bnAverages.updateWith(secBnResult, 0.99));
            return secBnResult.output;
        } else {
            return secRelu;
        }
    }

    private Tensor fullyConnectedLayer(Random rnd, Tensor inputs, RunMode runMode) {
        Tensor flattened = flatten(inputs);

        Tensor hidden_w = matmul(flattened, fullW);
        Tensor hiddenOut = cfg.useBatchNormalization ?
                hidden_w : add(hidden_w, fullB);
        Tensor hiddenRelu = relu(hiddenOut);

        Tensor hiddenFinal;
        if (cfg.useBatchNormalization) {
            BatchNormResult fullBnResult = batchNorm(hiddenRelu, full_bn_beta, full_bn_gamma, full_bnAverages, runMode);
            additionalUpdatesOnWeightUpdate.add(() -> {
                this.full_bnAverages = this.full_bnAverages.updateWith(fullBnResult, 0.99);
            });
            hiddenFinal = fullBnResult.output;
        } else {
            hiddenFinal = hiddenRelu;
        }

        return dropout(hiddenFinal, rnd, cfg.dropoutKeep, runMode);
    }

    private Tensor finalOutputLayer(Tensor inputs) {
        Tensor y_w = matmul(inputs, finalW);

        return add(y_w, finalB);
    }

    public List<TArray> getGradients() {
        List<Tensor> params = getParams();

        List<TArray> grads = new ArrayList<>();
        for (Tensor p : params) {
            grads.add(p.getGradient());
        }
        return grads;
    }

    public void resetGradients() {
        List<Tensor> params = getParams();
        for (Tensor p : params) {
            p.resetGradient();
        }
    }

    public Tensor getParam(int idx) {
        return getParams().get(idx);
    }

    public MNISTConvModel copy() {
        return new MNISTConvModel(this);
    }

    public static class Config {
        private final int firstConvChannels;
        private final int secondConvChannels;
        private final int fullyConnectedSize;
        private final double l2Lambda;
        private final long weightInitRandomSeed;
        private final boolean useBatchNormalization;
        private final double dropoutKeep;

        private Config(int firstConvChannels, int secondConvChannels, int fullyConnectedSize, double l2Lambda, long weightInitRandomSeed, boolean useBatchNormalization, double dropoutKeep) {
            this.firstConvChannels = firstConvChannels;
            this.secondConvChannels = secondConvChannels;
            this.fullyConnectedSize = fullyConnectedSize;
            this.l2Lambda = l2Lambda;
            this.weightInitRandomSeed = weightInitRandomSeed;
            this.useBatchNormalization = useBatchNormalization;
            this.dropoutKeep = dropoutKeep;
        }

        public static class Builder {
            private int firstConvChannels = 4;
            private int secondConvChannels = 8;
            private int fullyConnectedSize = 32;
            private double l2Lambda = 0.0;
            private double dropoutKeep = 0.5;
            private long weightInitRandomSeed = 4;
            private boolean useBatchNormalization = true;

            public Builder firstConvChannels(int firstConvChannels) {
                this.firstConvChannels = firstConvChannels;
                return this;
            }

            public Builder secondConvChannels(int secondConvChannels) {
                this.secondConvChannels = secondConvChannels;
                return this;
            }

            public Builder fullyConnectedSize(int fullyConnectedSize) {
                this.fullyConnectedSize = fullyConnectedSize;
                return this;
            }

            public Builder l2Lambda(double l2Lambda) {
                this.l2Lambda = l2Lambda;
                return this;
            }

            public Builder dropoutKeep(double dropoutKeep) {
                this.dropoutKeep = dropoutKeep;
                return this;
            }

            public Builder weightInitRandomSeed(long weightInitRandomSeed) {
                this.weightInitRandomSeed = weightInitRandomSeed;
                return this;
            }

            public Builder useBatchNormalization(boolean useBatchNormalization) {
                this.useBatchNormalization = useBatchNormalization;
                return this;
            }

            public static Builder cfgBuilder() {
                return new Builder();
            }

            public Config build() {
                return new Config(firstConvChannels, secondConvChannels, fullyConnectedSize,
                        l2Lambda, weightInitRandomSeed, useBatchNormalization, dropoutKeep);
            }
        }
    }

    public static class TrainStats {
        private double accTotal;
        private double costTotal;
        private double costL2Total;
        private int iterations;

        public void accumulate(double cost, double l2Cost, double accuracy) {
            costTotal += cost;
            costL2Total += l2Cost;
            accTotal += accuracy;
            iterations++;
        }

        @Override
        public String toString() {
            return "TrainStats{" +
                    "iterations=" + iterations +
                    ", accuracy=" + (accTotal / iterations) +
                    ", costTotal=" + (costTotal / iterations) +
                    ", costL2Total=" + (costL2Total / iterations) +
                    '}';
        }
    }

}
