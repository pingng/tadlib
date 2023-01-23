package com.codeberry.tadlib.example.mnist;

import com.codeberry.tadlib.array.Comparison;
import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.array.TArrayFactory;
import com.codeberry.tadlib.example.TrainingData;
import com.codeberry.tadlib.memorymanagement.DisposalRegister;
import com.codeberry.tadlib.nn.model.Model;
import com.codeberry.tadlib.nn.model.ModelFactory;
import com.codeberry.tadlib.provider.java.NDArray;
import com.codeberry.tadlib.provider.java.ValueUpdate;
import com.codeberry.tadlib.tensor.OpsExtended;
import com.codeberry.tadlib.tensor.Tensor;
import com.codeberry.tadlib.util.TrainingDataUtils;

import java.util.*;

import static com.codeberry.tadlib.array.TArrayFactory.onesShaped;
import static com.codeberry.tadlib.example.mnist.MNISTLoader.IMAGE_SIZE;
import static com.codeberry.tadlib.example.mnist.MNISTLoader.OUTPUTS;
import static com.codeberry.tadlib.nn.loss.L2Loss.l2LossOf;
import static com.codeberry.tadlib.provider.ProviderStore.array;
import static com.codeberry.tadlib.provider.ProviderStore.shape;
import static com.codeberry.tadlib.provider.java.NDArray.DimKeepRemove.KEEP_DIM;
import static com.codeberry.tadlib.provider.java.ValueUpdate.fromIndices;
import static com.codeberry.tadlib.tensor.Ops.*;
import static com.codeberry.tadlib.tensor.Tensor.TensorFactories.*;
import static com.codeberry.tadlib.tensor.Tensor.constant;
import static com.codeberry.tadlib.util.ReflectionUtils.*;
import static java.lang.Math.*;
import static java.util.Arrays.asList;
import static java.util.stream.Collectors.*;

/**
 * A hardcoded model using convolutions and attention.
 */
public class FixedMNISTConvAttentionModel implements Model {
    public static final int POS_CHANNELS = 32;

    private final Factory cfg;
    private final BatchNormLayer bnInput;

    private final Tensor w0;
    private final Tensor skipW0;
    private final BatchNormLayer bn0;

    private final Tensor w1;
    private final BatchNormLayer bn1;

    private final Tensor w2;
    private final Tensor skipW2;
    private final BatchNormLayer bn2;

    private final Tensor finalW;
    private final Tensor finalB;


    private final BlockDropoutLayer blockDropoutLayer;

    private final AttentionLayer[] attentionLayers;

    private static class BatchNormLayer {
        private final Tensor sec_bn_beta;
        private final Tensor sec_bn_gamma;
        public OpsExtended.BatchNormRunningAverages bnAverages = new OpsExtended.BatchNormRunningAverages();

        BatchNormLayer(int channels) {
            this.sec_bn_beta = zeros(shape(channels));
            this.sec_bn_gamma = ones(shape(channels));
        }

        Tensor batchNorm(List<Runnable> trainingTasks, RunMode runMode, Tensor y) {
            OpsExtended.BatchNormResult result = OpsExtended.batchNorm(y, sec_bn_beta, sec_bn_gamma, bnAverages, runMode);
            trainingTasks.add(() -> this.bnAverages.updateWith(result, 0.99));
            y = result.output;
            return y;
        }
    }

    public FixedMNISTConvAttentionModel(Factory cfg) {
        this.cfg = cfg;

        this.bnInput = new BatchNormLayer(1);

        Random r = new Random(cfg.weightInitRandomSeed);
        this.w0 = randomWeight(r, shape(3, 3, 1, cfg.firstConvChannels));
        this.skipW0 = randomWeight(r, shape(1, 1, 1, cfg.firstConvChannels));
        this.bn0 = new BatchNormLayer(cfg.firstConvChannels);

        this.w1 = randomWeight(r, shape(3, 3, cfg.firstConvChannels, cfg.firstConvChannels));
        this.bn1 = new BatchNormLayer(cfg.firstConvChannels);

        this.w2 = randomWeight(r, shape(3, 3, cfg.firstConvChannels, cfg.secondConvChannels));
        this.skipW2 = randomWeight(r, shape(1, 1, cfg.firstConvChannels, cfg.secondConvChannels));

        this.bn2 = new BatchNormLayer(cfg.secondConvChannels);

        this.attentionLayers = new AttentionLayer[cfg.attentionLayers];
        for (int i = 0; i < cfg.attentionLayers; i++) {
            this.attentionLayers[i] = new AttentionLayer(r, cfg.heads, POS_CHANNELS, IMAGE_SIZE / 2 / 2, cfg.secondConvChannels);
        }
        this.blockDropoutLayer = new BlockDropoutLayer(3);

        int size = IMAGE_SIZE / 2 / 2;
        int finalChans = cfg.secondConvChannels;
        this.finalW = randomWeight(r, shape(size * size * finalChans, OUTPUTS));
        this.finalB = zeros(shape(OUTPUTS));
    }

    @Override
    public String getTrainingLogText() {
        StringBuilder str = new StringBuilder();
        str.append("BatchNormData\n")
                .append("INP:").append(bnInput.bnAverages).append("\n")
                .append("BN0:").append(bn0.bnAverages).append("\n")
                .append("BN1:").append(bn1.bnAverages).append("\n")
                .append("BN2:").append(bn2.bnAverages).append("\n");
        for (int i = 0; i < attentionLayers.length; i++) {
            str.append("AttentionBn").append(i).append(":").append(attentionLayers[i].bn.bnAverages).append("\n");
        }
        return str.toString().trim();
    }

    public PredictionAndLosses calcCost(Random rnd,
                                        TrainingData.Batch trainingData, IterationInfo iterationInfo) {
        int actualBatchSize = trainingData.getBatchSize();

        List<Runnable> trainingTasks = new ArrayList<>();
        Tensor y = forward(rnd, trainingData.input, trainingTasks, RunMode.TRAINING, iterationInfo);

        Tensor totalSoftmaxCost = sumSoftmaxCrossEntropy(TrainingDataUtils.toOneHot(trainingData.output, 10), y);
        Tensor avgSoftmaxCost = div(totalSoftmaxCost, constant(actualBatchSize));

        List<Tensor> weights = new ArrayList<>(asList(w0, skipW0, w1, w2, skipW2, finalW));

        for (AttentionLayer al : attentionLayers) {
            for (AttentionLayer.Head head : al.heads) {
                weights.addAll(asList(head.attKeyW, head.attQueryW, head.attValueW));
            }
        }

        Tensor[] ws = weights.toArray(Tensor[]::new);
        Tensor l2Loss = cfg.l2Lambda <= 0 ? Tensor.ZERO :
                div(l2LossOf(cfg.l2Lambda,
                        ws), constant(actualBatchSize));

        Tensor totalLoss = add(avgSoftmaxCost, l2Loss);

        return new PredictionAndLosses(y, trainingTasks, totalLoss, l2Loss);
    }

    @Override
    public List<DisposalRegister.Disposable> getKeepInMemoryDisposables() {
        List<BatchNormLayer> batchNormLayers = getFieldValues(BatchNormLayer.class, this);
        List<DisposalRegister.Disposable> keepObjects = batchNormLayers.stream()
                .map(batchNormLayer -> batchNormLayer.bnAverages.getKeepInMemoryDisposables())
                .flatMap(Collection::stream)
                .collect(toList());

        for (AttentionLayer al : attentionLayers) {
            List<BatchNormLayer> alBns = getFieldValues(BatchNormLayer.class, al);
            for (BatchNormLayer alBn : alBns) {
                keepObjects.addAll(alBn.bnAverages.getKeepInMemoryDisposables());
            }
        }

        return keepObjects;
    }

    public List<Tensor> getParams() {
        List<Tensor> params = getFieldValues(Tensor.class, this);

        for (AttentionLayer al : attentionLayers) {
            params.addAll(getFieldValues(Tensor.class, al));
            List<BatchNormLayer> alBns = getFieldValues(BatchNormLayer.class, al);
            for (BatchNormLayer alBn : alBns) {
                params.add(alBn.sec_bn_beta);
                params.add(alBn.sec_bn_gamma);
            }
            for (AttentionLayer.Head head : al.heads) {
                params.addAll(head.getParams());
            }
        }

        List<BatchNormLayer> batchNormLayers = getFieldValues(BatchNormLayer.class, this);
        for (BatchNormLayer batchNormLayer : batchNormLayers) {
            params.addAll(getFieldValues(Tensor.class, batchNormLayer));
        }
        return params;
    }

    public Tensor predict(Tensor x_train, IterationInfo iterationInfo) {
        return forward(null, x_train, new ArrayList<>(), RunMode.INFERENCE, new IterationInfo(0, 0, 1));
    }

    private Tensor forward(Random rnd, Tensor inputs, List<Runnable> trainingTasks, RunMode runMode, IterationInfo iterationInfo) {
        Tensor y = inputs;

        y = bnInput.batchNorm(trainingTasks, runMode, y);
        y = blockDropoutLayer.forward(rnd, y, runMode, iterationInfo);

        Tensor skipped0 = conv2d(y, skipW0);
        y = convReluBn(y, w0, bn0, trainingTasks, runMode);
        y = convReluBn(y, w1, bn1, trainingTasks, runMode);
        y = add(skipped0, y);
        y = maxpool2d(y, 2);
        y = blockDropoutLayer.forward(rnd, y, runMode, iterationInfo);

        Tensor skipped2 = conv2d(y, skipW2);
        y = convReluBn(y, w2, bn2, trainingTasks, runMode);
        y = add(skipped2, y);
        y = maxpool2d(y, 2);
        y = blockDropoutLayer.forward(rnd, y, runMode, iterationInfo);

        Tensor finalSkipped = y;

        for (AttentionLayer attentionLayer : attentionLayers) {
            y = attentionLayer.forward(trainingTasks, runMode, y);
        }

        y = add(finalSkipped, y);

        return finalOutputLayer(y, rnd, runMode);
    }

    private static Tensor convReluBn(Tensor inputs, Tensor w, BatchNormLayer bn, List<Runnable> trainingTasks, RunMode runMode) {
        Tensor h_w = conv2d(inputs, w);
        return bn.batchNorm(trainingTasks, runMode, leakyRelu(h_w, 0.01));
    }

    private Tensor finalOutputLayer(Tensor inputs, Random rnd, RunMode runMode) {
        Shape shape = inputs.shape();
        Tensor flattened = reshape(inputs, shape.at(0), -1);

        Tensor y_w = matmul(dropout(flattened, rnd, cfg.dropoutKeep, runMode), finalW);

        return add(y_w, finalB);
    }

    public static class Factory implements ModelFactory {
        private final int firstConvChannels;
        private final int secondConvChannels;
        private final double l2Lambda;
        private final long weightInitRandomSeed;
        private final double dropoutKeep;
        private final int heads;
        private final int attentionLayers;

        private Factory(int firstConvChannels, int secondConvChannels, double l2Lambda, long weightInitRandomSeed, int heads, double dropoutKeep, int attentionLayers) {
            this.firstConvChannels = firstConvChannels;
            this.secondConvChannels = secondConvChannels;
            this.l2Lambda = l2Lambda;
            this.weightInitRandomSeed = weightInitRandomSeed;
            this.heads = heads;
            this.dropoutKeep = dropoutKeep;
            this.attentionLayers = attentionLayers;
        }

        @Override
        public Model createModel() {
            return new FixedMNISTConvAttentionModel(this);
        }

        public static class Builder {
            private int firstConvChannels = 4;
            private int secondConvChannels = 8;
            private double l2Lambda = 0.0;
            private double dropoutKeep = 0.5;
            private long weightInitRandomSeed = 4;
            private int heads = 8;
            private int attentionLayers = 1;

            public Builder firstConvChannels(int firstConvChannels) {
                this.firstConvChannels = firstConvChannels;
                return this;
            }

            public Builder secondConvChannels(int secondConvChannels) {
                this.secondConvChannels = secondConvChannels;
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

            public Builder multiAttentionHeads(int heads) {
                this.heads = heads;
                return this;
            }

            public Builder attentionLayers(int attentionLayers) {
                this.attentionLayers = attentionLayers;
                return this;
            }

            public static Builder factoryBuilder() {
                return new Builder();
            }

            public Factory build() {
                return new Factory(firstConvChannels, secondConvChannels,
                        l2Lambda, weightInitRandomSeed, heads, dropoutKeep, attentionLayers);
            }
        }
    }

    // https://papers.nips.cc/paper/2018/file/7edcfb2d8f6a659ef4cd1e6c9b6d7079-Paper.pdf
    private static class BlockDropoutLayer {
        private static final double DROP_KEEP_START = 0.999;
        private static final double[] DROP_KEEP_SLOW_DOWN = {0.9, 0.8, 0.75};
        private static final double[] DROP_KEEP_REDUCE = {0.992, 0.999, 0.9995};
        private static final double DROP_KEEP_MIN = DROP_KEEP_SLOW_DOWN[DROP_KEEP_SLOW_DOWN.length - 1];

        private final int blockSize;

        private volatile int alreadyUpdatedAtEpoch = -1;
        private volatile double dropKeep = DROP_KEEP_START;

        private BlockDropoutLayer(int blockSize) {
            this.blockSize = blockSize;
        }

        Tensor forward(Random r, Tensor input, RunMode mode, IterationInfo iterationInfo) {
            if (mode == RunMode.TRAINING) {
                Shape inputShape = input.shape();
                double featureSize = inputShape.at(-2);

                if (iterationInfo.hasPrevEpochTrainInfo() &&
                        iterationInfo.batchIndex == 0 && alreadyUpdatedAtEpoch != iterationInfo.epoch) {
                    TrainInfo trainInfo = iterationInfo.prevEpochTrainInfo;
                    if (trainInfo.isValid() && trainInfo.trainingIsMoreAccurateThanTesting()) {
                        double d = dropKeep;

                        d *= getReduceRate(d);
                        d = max(d, DROP_KEEP_MIN);
                        System.out.println("New block drop keep: " + dropKeep + " -> " + d);
                        dropKeep = d;
                        alreadyUpdatedAtEpoch = iterationInfo.epoch;
                    }
                }

                double validRegionSize = featureSize - blockSize + 1;
                double gamma = ((1.0 - dropKeep) / (blockSize * blockSize)) * ((featureSize * featureSize) / (validRegionSize * validRegionSize));

                List<ValueUpdate> updates = createZeroUpdates(r, inputShape, gamma, this.blockSize);

                if (!updates.isEmpty()) {
                    int channels = inputShape.at(-1);
                    double area2d = inputShape.at(-3) * inputShape.at(-2);
                    NDArray blockFilter = createSameChannelFilter(channels);

                    NDArray ones = TArrayFactory.ones(inputShape);
                    NDArray zeroSeeds = ones.withUpdates(updates);
//                    debugPrintMask("Seed", zeroSeeds, true);
                    // zeroSeeds.shape = (-1, h, w, c) ==> (-1, c, h, w)
                    NDArray floatingPointMask = zeroSeeds.conv2d(blockFilter);
//                    debugPrintMask("FloatMask", floatingPointMask, true);
                    NDArray zerosForCompare = array(0.0);
                    NDArray mask = floatingPointMask.compare(zerosForCompare, Comparison.greaterThan(), 1.0, 0.0);
                    NDArray oneCount = mask.sum(KEEP_DIM, -3, -2);
//                    System.out.println(oneCount);
//                    debugPrintMask("One count", oneCount, true);
//                    System.out.println(mask);
//                    debugPrintMask("Mask", mask, true);
                    NDArray scaledMask = mask.mul(area2d).div(oneCount);
//                    debugPrintMask("Scaled Mask", scaledMask, true);

                    Tensor finalMask = constant(scaledMask);

                    return mul(input, finalMask);
                }
            }
            return input;
        }

        private static double getReduceRate(double d) {
            for (int i = 0; i < DROP_KEEP_SLOW_DOWN.length; i++) {
                if (d > DROP_KEEP_SLOW_DOWN[i]) {
                    return DROP_KEEP_REDUCE[i];
                }
            }
            return 1.0;
        }

        /**
         * @return conv filter that has ones only if inChannelId==outChannelId
         */
        private NDArray createSameChannelFilter(int channels) {
            NDArray blockFilterOnes = TArrayFactory.onesShaped(blockSize, blockSize, channels, channels);
            NDArray identity = TArrayFactory.onesShaped(channels).diag();

            return blockFilterOnes.mul(identity);
        }

        private static void debugPrintMask(String title, NDArray mask, boolean firstOnly) {
            Shape shape = mask.shape;
            if (shape.getDimCount() != 4) {
                throw new RuntimeException("Expected 4 dims");
            }
            int batchSize = shape.at(0);
            int height = shape.at(1);
            int width = shape.at(2);
            int channels = shape.at(3);

            System.out.println("------ " + title + " ------");
            double[][][][] vals = (double[][][][]) mask.toDoubles();
            for (int example = 0; example < batchSize; example++) {
                System.out.println("--- Ex " + example);
                for (int c = 0; c < channels; c++) {
                    System.out.println("- chan" + c);
                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width; x++) {
                            double v = vals[example][y][x][c];
                            if (v >= 0) {
                                System.out.print((int) v);
                            } else {
                                System.out.print("-");
                            }
                        }
                        System.out.println();
                    }
                    System.out.println();
                    if (firstOnly) {
                        return;
                    }
                }
            }

        }

        private static void debugPrintFilter(String title, NDArray mask, boolean firstOnly) {
            Shape shape = mask.shape;
            if (shape.getDimCount() != 4) {
                throw new RuntimeException("Expected 4 dims");
            }
            int height = shape.at(0);
            int width = shape.at(1);
            int inChans = shape.at(2);
            int outChans = shape.at(3);

            System.out.println("------ " + title + " ------");
            double[][][][] vals = (double[][][][]) mask.toDoubles();
            for (int out = 0; out < outChans; out++) {
                System.out.println("--- Out " + out);
                for (int in = 0; in < inChans; in++) {
                    System.out.println("- In " + in);
                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width; x++) {
                            double v = vals[y][x][in][out];
                            if (v >= 0) {
                                System.out.print((int) v);
                            } else {
                                System.out.print("-");
                            }
                        }
                        System.out.println();
                    }
                    System.out.println();
                    if (firstOnly) {
                        return;
                    }
                }
            }

        }

        private static List<ValueUpdate> createZeroUpdates(Random r, Shape inputShape, double gamma, int blockSize) {
            List<ValueUpdate> ret = new ArrayList<>();

            fillZeroUpdates(r, inputShape, gamma, blockSize, ret, inputShape.newIndexArray(), 0);

            return ret;
        }

        private static void fillZeroUpdates(Random r, Shape shape, double gamma, int blockSize, List<ValueUpdate> updates, int[] indices, int dim) {
            int len = shape.at(dim);
            int from, to;

            boolean isLastDimension = (dim == indices.length - 1);
            boolean isIn2dPlane = (dim == indices.length - 3) || (dim == indices.length - 2);
            if (isIn2dPlane) {
                from = blockSize / 2;
                to = len - blockSize / 2;
            } else {
                from = 0;
                to = len;
            }

            for (int i = from; i < to; i++) {
                indices[dim] = i;

                if (isLastDimension) {
                    if (r.nextDouble() < gamma) {
                        updates.add(fromIndices(-Float.MAX_VALUE, shape, indices));
                    }
                } else {
                    fillZeroUpdates(r, shape, gamma, blockSize, updates, indices, dim + 1);
                }
            }
        }
    }

    private static class AttentionLayer {
        private final Head[] heads;
        private final NDArray singleExample;
        private final BatchNormLayer bn;
        private final Tensor w;

        public AttentionLayer(Random r, int heads, int positionChannels, int size, int inOutChannels) {
            int headOutChannels = inOutChannels / heads;
            if ((inOutChannels % headOutChannels) != 0) {
                throw new IllegalArgumentException("OutputChannels not divisible by heads");
            }

            this.heads = new Head[heads];
            for (int i = 0; i < heads; i++) {
                this.heads[i] = new Head(r, positionChannels, inOutChannels, headOutChannels);
            }

            this.w = randomWeight(r, shape(3, 3, inOutChannels, inOutChannels));

            this.bn = new BatchNormLayer(inOutChannels);

            this.singleExample = PositionEncoder.create2dPositionEncoding(positionChannels, size);
        }

        public Tensor forward(List<Runnable> trainingTasks, RunMode runMode, Tensor inputs) {
            Tensor positionEncoding = getPosEncodingForBatch(inputs.shape().at(0));
            Tensor withPosEnc = concat(-1, inputs, positionEncoding);

            Tensor[] headOuts = new Tensor[heads.length];
            for (int i = 0; i < heads.length; i++) {
                headOuts[i] = heads[i].forward(withPosEnc);
            }
            Tensor allHeads = concat(-1, headOuts);

            Tensor skipSum = add(inputs, conv2d(allHeads, w));
            return bn.batchNorm(trainingTasks, runMode, skipSum);
        }

        private Tensor getPosEncodingForBatch(int batchSize) {
            NDArray batchBroadcast = TArrayFactory.onesShaped(batchSize, 1, 1, 1);
            NDArray repeatedForWholeBatch = batchBroadcast.mul(singleExample);

            return constant(repeatedForWholeBatch);
        }

        static class Head {
            private final Tensor attKeyW;
            private final Tensor attQueryW;
            private final Tensor attValueW;
            private final Tensor attKeyB;
            private final Tensor attQueryB;
            private final Tensor attValueB;

            Head(Random r, int posChannels, int inputChannels, int outputChannels) {
                Shape wShape = shape(3, 3, inputChannels + posChannels, outputChannels);
                Shape bShape = shape(outputChannels);
                this.attKeyW = randomWeight(r, wShape);
                this.attQueryW = randomWeight(r, wShape);
                this.attValueW = randomWeight(r, wShape);
                this.attKeyB = randomWeight(r, bShape);
                this.attQueryB = randomWeight(r, bShape);
                this.attValueB = randomWeight(r, bShape);
            }

            public Collection<? extends Tensor> getParams() {
                return asList(attKeyW, attQueryW, attValueW,
                        attKeyB, attQueryB, attValueB);
            }

            Tensor forward(Tensor inputs) {
                Shape shape = inputs.shape();
                int examples = shape.at(0);
                int h = shape.at(1);
                int w = shape.at(2);
                int chan = attKeyW.shape().at(3);
                int area = h * w;

                Tensor attKey = add(conv2d(inputs, attKeyW), attKeyB);
                Tensor attQuery = add(conv2d(inputs, attQueryW), attQueryB);
                Tensor chanRowSequence = reshape(attKey, examples, 1, area, chan);
                Tensor chanColSequence = reshape(attQuery, examples, area, chan, 1);
                Tensor matmul = matmul(chanRowSequence, chanColSequence);
                Tensor preSoftmax = reshape(matmul, examples, area, area);
                Tensor softmax = softmax(preSoftmax);
                Tensor softmaxShaped = reshape(softmax, examples, area, area, 1);
                Tensor attValueRaw = add(conv2d(inputs, attValueW), attValueB);
                Tensor attValue = reshape(attValueRaw, examples, 1, area, chan);

                Tensor scaled = mul(attValue, softmaxShaped);
                Tensor scaleSummed = sum(scaled, 2);
                return reshape(scaleSummed, examples, h, w, chan);
            }
        }
    }

    abstract static class PositionEncoder {
        public static NDArray create2dPositionEncoding(int totalChannels, int size) {
            int positions = size;

            NDArray encoding = createPositionEncoding(totalChannels / 2, positions);
    //        System.out.println("encoding = " + encoding);

            NDArray horizontal = encoding.mul(onesShaped(positions, 1, 1));
    //        double[][][] repeated = (double[][][]) horizontal.toDoubles();
    //        System.out.println("matmul = " + deepToString(repeated[0]));
    //        System.out.println("matmul = " + deepToString(repeated[1]));
    //        System.out.println("matmul = " + deepToString(repeated[2]));
    //        System.out.println("matmul = " + deepToString(repeated[3]));

    //        System.out.println("-- Vert");
            NDArray vertical = horizontal.transpose(1, 0, 2);
    //        double[][][] vert = (double[][][]) vertical.toDoubles();
    //        System.out.println(deepToString(vert[0]));
    //        System.out.println(deepToString(vert[1]));
    //        System.out.println(deepToString(vert[2]));

            NDArray both = horizontal.concat(vertical, -1);
            return both;
        }

        private static NDArray createPositionEncoding(int channels, int positions) {
            double[][] posEnc = new double[positions][channels];
            for (int p = 0; p < posEnc.length; p++) {
                double[] chans = posEnc[p];
                for (int c = 0; c < chans.length; c++) {
                    double angleRate = 1 / pow(10000, (2 * floor(c/ 2.0)) / channels);
                    double angle = angleRate * p;

                    chans[c] = ((c & 1) == 0 ?
                            sin(angle) :
                            cos(angle));
                }
            }
            return array(posEnc);
        }
    }
}
