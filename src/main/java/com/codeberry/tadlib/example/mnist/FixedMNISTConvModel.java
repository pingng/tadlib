package com.codeberry.tadlib.example.mnist;

import com.codeberry.tadlib.memorymanagement.DisposalRegister;
import com.codeberry.tadlib.nn.model.Model;
import com.codeberry.tadlib.nn.model.ModelFactory;
import com.codeberry.tadlib.tensor.Tensor;
import com.codeberry.tadlib.util.ReflectionUtils;
import com.codeberry.tadlib.util.TrainingDataUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static com.codeberry.tadlib.example.TrainingData.*;
import static com.codeberry.tadlib.example.mnist.MNISTLoader.*;
import static com.codeberry.tadlib.nn.loss.L2Loss.l2LossOf;
import static com.codeberry.tadlib.provider.ProviderStore.shape;
import static com.codeberry.tadlib.tensor.Ops.*;
import static com.codeberry.tadlib.tensor.OpsExtended.*;
import static com.codeberry.tadlib.tensor.Tensor.TensorFactories.*;
import static com.codeberry.tadlib.tensor.Tensor.constant;

/**
 * A hardcoded model using convolutions.
 */
public class FixedMNISTConvModel implements Model {
    private final Factory cfg;

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

    public final BatchNormRunningAverages secondConvBnAverages = new BatchNormRunningAverages();
    public final BatchNormRunningAverages fullyConBnAverages = new BatchNormRunningAverages();

    public FixedMNISTConvModel(Factory cfg) {
        this.cfg = cfg;

        Random r = new Random(cfg.weightInitRandomSeed);
        this.w = randomWeight(r, shape(3, 3, 1, cfg.firstConvChannels));
        this.b = zeros(shape(cfg.firstConvChannels));

        this.sec_w = randomWeight(r, shape(3, 3, cfg.firstConvChannels, cfg.secondConvChannels));

        if (cfg.useBatchNormalization) {
            this.sec_b = null;
            this.fullB = null;

            this.sec_bn_beta = zeros(shape(cfg.secondConvChannels));
            this.sec_bn_gamma = ones(shape(cfg.secondConvChannels));
            this.full_bn_beta = zeros(shape(cfg.fullyConnectedSize));
            this.full_bn_gamma = ones(shape(cfg.fullyConnectedSize));
        } else {
            this.sec_b = zeros(shape(cfg.secondConvChannels));
            this.fullB = zeros(shape(cfg.fullyConnectedSize));

            this.sec_bn_beta = null;
            this.sec_bn_gamma = null;
            this.full_bn_beta = null;
            this.full_bn_gamma = null;
        }

        int hiddenSize = IMAGE_SIZE / 2 / 2;
        this.fullW = randomWeight(r,
                shape(hiddenSize * hiddenSize * cfg.secondConvChannels,
                        cfg.fullyConnectedSize));

        this.finalW = randomWeight(r, shape(cfg.fullyConnectedSize, OUTPUTS));
        this.finalB = zeros(shape(OUTPUTS));
    }

    @Override
    public String getTrainingLogText() {
        return "secondConvBnAverages:\n" + secondConvBnAverages + "\n" +
                "fullyConBnAverages:\n" + fullyConBnAverages;
    }

    public PredictionAndLosses calcCost(Random rnd, Batch trainingData, IterationInfo iterationInfo) {
        int actualBatchSize = trainingData.getBatchSize();

        List<Runnable> trainingTasks = new ArrayList<>();
        Tensor y = forward(rnd, trainingData.input, trainingTasks, RunMode.TRAINING);

        Tensor totalSoftmaxCost = sumSoftmaxCrossEntropy(TrainingDataUtils.toOneHot(trainingData.output, 10), y);
        Tensor avgSoftmaxCost = div(totalSoftmaxCost, constant(actualBatchSize));

        Tensor l2Loss = cfg.l2Lambda <= 0 ? Tensor.ZERO :
                div(l2LossOf(cfg.l2Lambda,
                        w, sec_w, fullW, finalW), constant(actualBatchSize));

        Tensor totalLoss = add(avgSoftmaxCost, l2Loss);

        return new PredictionAndLosses(y, trainingTasks, totalLoss, l2Loss);
    }

    @Override
    public List<DisposalRegister.Disposable> getKeepInMemoryDisposables() {
        List<DisposalRegister.Disposable> r = new ArrayList<>();
        r.addAll(secondConvBnAverages.getKeepInMemoryDisposables());
        r.addAll(fullyConBnAverages.getKeepInMemoryDisposables());
        return r;
    }

    public List<Tensor> getParams() {
        return ReflectionUtils.getFieldValues(Tensor.class, this);
    }

    public Tensor predict(Tensor x_train, IterationInfo iterationInfo) {
        return forward(null, x_train, new ArrayList<>(), RunMode.INFERENCE);
    }

    private Tensor forward(Random rnd, Tensor inputs, List<Runnable> trainingTasks, RunMode runMode) {
        Tensor firstLayerOut = firstConvLayer(inputs);
        Tensor secondLayerOut = secondConvLayer(runMode, firstLayerOut, trainingTasks);
        Tensor fullyLayerOut = fullyConnectedLayer(rnd, secondLayerOut, runMode, trainingTasks);
        return finalOutputLayer(fullyLayerOut);
    }

    private Tensor firstConvLayer(Tensor inputs) {
        Tensor h_w = conv2d(inputs, w);
        Tensor h = add(h_w, b);
        Tensor maxed = maxpool2d(h, 2);
        return leakyRelu(maxed, 0.01);
    }

    private Tensor secondConvLayer(RunMode runMode, Tensor inputs, List<Runnable> trainingTasks) {
        Tensor sec_h_w = conv2d(inputs, sec_w);
        Tensor secOut = cfg.useBatchNormalization ?
                sec_h_w : add(sec_h_w, sec_b);

        Tensor sec_maxed = maxpool2d(secOut, 2);
        Tensor secRelu = leakyRelu(sec_maxed, 0.01);

        if (cfg.useBatchNormalization) {
            BatchNormResult secBnResult = batchNorm(secRelu, sec_bn_beta, sec_bn_gamma, secondConvBnAverages, runMode);
            trainingTasks.add(() -> this.secondConvBnAverages.updateWith(secBnResult, 0.99));
            return secBnResult.output;
        } else {
            return secRelu;
        }
    }

    private Tensor fullyConnectedLayer(Random rnd, Tensor inputs, RunMode runMode, List<Runnable> trainingTasks) {
        Tensor flattened = flatten(inputs);

        Tensor hidden_w = matmul(flattened, fullW);
        Tensor hiddenOut = cfg.useBatchNormalization ?
                hidden_w : add(hidden_w, fullB);
        Tensor hiddenRelu = leakyRelu(hiddenOut, 0.01);

        Tensor hiddenFinal;
        if (cfg.useBatchNormalization) {
            BatchNormResult fullBnResult = batchNorm(hiddenRelu, full_bn_beta, full_bn_gamma, fullyConBnAverages, runMode);
            trainingTasks.add(() -> this.fullyConBnAverages.updateWith(fullBnResult, 0.99));
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

    public static class Factory implements ModelFactory {
        private final int firstConvChannels;
        private final int secondConvChannels;
        private final int fullyConnectedSize;
        private final double l2Lambda;
        private final long weightInitRandomSeed;
        private final boolean useBatchNormalization;
        private final double dropoutKeep;

        private Factory(int firstConvChannels, int secondConvChannels, int fullyConnectedSize, double l2Lambda, long weightInitRandomSeed, boolean useBatchNormalization, double dropoutKeep) {
            this.firstConvChannels = firstConvChannels;
            this.secondConvChannels = secondConvChannels;
            this.fullyConnectedSize = fullyConnectedSize;
            this.l2Lambda = l2Lambda;
            this.weightInitRandomSeed = weightInitRandomSeed;
            this.useBatchNormalization = useBatchNormalization;
            this.dropoutKeep = dropoutKeep;
        }

        @Override
        public Model createModel() {
            return new FixedMNISTConvModel(this);
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

            public static Builder factoryBuilder() {
                return new Builder();
            }

            public Factory build() {
                return new Factory(firstConvChannels, secondConvChannels, fullyConnectedSize,
                        l2Lambda, weightInitRandomSeed, useBatchNormalization, dropoutKeep);
            }
        }
    }
}
