package com.codeberry.tadlib.nn.model.layer;

import com.codeberry.tadlib.array.Comparison;
import com.codeberry.tadlib.array.NDArray;
import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.array.TArrayFactory;
import com.codeberry.tadlib.nn.model.Model;
import com.codeberry.tadlib.tensor.Tensor;
import com.codeberry.tadlib.util.Interpolation;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static com.codeberry.tadlib.array.Comparison.*;
import static com.codeberry.tadlib.array.NDArray.DimKeepRemove.KEEP_DIM;
import static com.codeberry.tadlib.array.NDArray.ValueUpdate.fromIndices;
import static com.codeberry.tadlib.nn.model.layer.Layer.ForwardResult.result;
import static com.codeberry.tadlib.provider.ProviderStore.array;
import static com.codeberry.tadlib.tensor.Ops.RunMode;
import static com.codeberry.tadlib.tensor.Ops.mul;
import static com.codeberry.tadlib.tensor.Tensor.constant;
import static java.lang.Math.max;

/**
 * Reference: https://papers.nips.cc/paper/2018/file/7edcfb2d8f6a659ef4cd1e6c9b6d7079-Paper.pdf
 */
public class BlockDropOutLayer implements Layer {
    private final int blockSize;
    private final Shape outputShape;
    private final Interpolation dropKeepInterpolation;

    public BlockDropOutLayer(Shape inputShape, Builder params) {
        this.outputShape = inputShape;
        this.dropKeepInterpolation = new Interpolation(params.valueStart, params.valueEnd,
                params.rangeStart, params.rangeEnd);
        this.blockSize = params.blockSize;
    }

    @Override
    public ForwardResult forward(Random rnd, Tensor inputs, RunMode runMode, Model.IterationInfo iterationInfo) {
        if (runMode == RunMode.TRAINING) {
            Shape inputShape = inputs.getShape();
            double featureSize = getFeatureSize(inputShape);

            double gamma = calcGamma(runMode, iterationInfo, featureSize);

            // Pick position as drop block 'seeds'
            List<NDArray.ValueUpdate> updates = createRandomUpdatesBasedOnGamma(rnd, inputShape, gamma, this.blockSize, -Float.MAX_VALUE);

            if (!updates.isEmpty()) {
                int channels = inputShape.at(-1);
                double area2d = inputShape.at(-3) * inputShape.at(-2);
                NDArray blockFilter = createIdentityChannelFilterOfSize(channels, blockSize);

                NDArray ones = TArrayFactory.ones(inputShape);
                NDArray onesWithSeeds = ones.withUpdates(updates);

                NDArray maskOfOnesAndNegValues = onesWithSeeds.conv2d(blockFilter);

                NDArray zero = array(0.0);
                NDArray maskOfOnesAndZeros = maskOfOnesAndNegValues.compare(zero, greaterThan(), 1.0, 0.0);
                NDArray onesInEach2dChannel = maskOfOnesAndZeros.sum(KEEP_DIM, -3, -2);
                NDArray scaledMask = maskOfOnesAndZeros.mul(area2d).div(onesInEach2dChannel);

                Tensor finalMask = constant(scaledMask);

                return result(mul(inputs, finalMask));
            }
        }


        return result(inputs);
    }

    private double calcGamma(RunMode runMode, Model.IterationInfo iterationInfo, double featureSize) {
        double dropKeep = dropKeepInterpolation.interpolate(iterationInfo.epoch);
        if (iterationInfo.batchIndex == 0 && runMode == RunMode.TRAINING) {
            System.out.println(getClass().getSimpleName() + ", " + outputShape + ": dropKeep=" + dropKeep);
        }
        double validRegionSize = featureSize - blockSize + 1;

        return ((1. - dropKeep) / (blockSize * blockSize)) * ((featureSize * featureSize) / (validRegionSize * validRegionSize));
    }

    private static double getFeatureSize(Shape inputShape) {
        double featureHeight = inputShape.at(-3);
        double featureWidth = inputShape.at(-2);

        if (featureWidth != featureHeight) {
            throw new IllegalArgumentException("Expects 2d height == width (dimension -3 & -2, respectively): " +
                    featureHeight + ", " + featureWidth);
        }

        return featureWidth;
    }

    private static List<NDArray.ValueUpdate> createRandomUpdatesBasedOnGamma(Random r, Shape inputShape, double gamma, int blockSize, double value) {
        List<NDArray.ValueUpdate> ret = new ArrayList<>();

        fillUpdates(r, inputShape, gamma, blockSize, ret, inputShape.newIndexArray(), 0, value);

        return ret;
    }

    private static void fillUpdates(Random r, Shape shape, double gamma, int blockSize, List<NDArray.ValueUpdate> updates, int[] indices, int dim, double value) {
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
                    updates.add(fromIndices(value, shape, indices));
                }
            } else {
                fillUpdates(r, shape, gamma, blockSize, updates, indices, dim + 1, value);
            }
        }
    }

    /**
     * @return conv filter that has ones only if inChannelId==outChannelId
     */
    private static NDArray createIdentityChannelFilterOfSize(int channels, int blockSize) {
        NDArray blockFilterOnes = TArrayFactory.onesShaped(blockSize, blockSize, channels, channels);
        NDArray identity = TArrayFactory.onesShaped(channels).diag();

        return blockFilterOnes.mul(identity);
    }

    @Override
    public Shape getOutputShape() {
        return outputShape;
    }

    public static class Builder implements LayerBuilder {
        private int blockSize;
        private double valueStart;
        private double valueEnd;
        private int rangeStart;
        private int rangeEnd;

        @Override
        public Layer build(Random rnd, Shape inputShape) {
            return new BlockDropOutLayer(inputShape, this);
        }

        public static Builder blockDropout() {
            return new Builder();
        }

        public Builder blockSize(int blockSize) {
            this.blockSize = blockSize;
            return this;
        }

        public Builder dropKeepRange(double start, double end) {
            this.valueStart = start;
            this.valueEnd = end;
            return this;
        }

        public Builder epochRange(int start, int end) {
            this.rangeStart = start;
            this.rangeEnd = end;
            return this;
        }
    }
}
