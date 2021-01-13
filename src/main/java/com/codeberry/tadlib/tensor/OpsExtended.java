package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.array.NDArray;
import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.memorymanagement.DisposalRegister;

import java.util.List;

import static com.codeberry.tadlib.memorymanagement.DisposalRegister.*;
import static java.util.Arrays.asList;

public abstract class OpsExtended {
    public static int guessParamLength(Shape shape) {
        int dimCount = shape.getDimCount();
        if (dimCount == 2 || dimCount == 4) {
            int channels = shape.at(-1);

            return channels;
        }
        throw new IllegalArgumentException("Valid dims are 2 an 4");
    }

    public static BatchNormResult batchNorm(Tensor input, Tensor beta, Tensor gamma, BatchNormRunningAverages averages, Ops.RunMode runMode) {
        Shape shape = input.getVals().getShape();
        if (shape.getDimCount() != 2 && shape.getDimCount() != 4) {
            throw new IllegalArgumentException("Valid dims are 2 an 4");
        }

        if (shape.getDimCount() == 4) {
            int channels = shape.at(-1);

            Tensor mean;
            Tensor _diff;
            Tensor variance;

            if (runMode == Ops.RunMode.TRAINING) {
                mean = Ops.mean(input, 0, 1, 2);
                _diff = Ops.sub(input, Ops.reshape(mean, 1, 1, 1, channels));
                Tensor _sqrDiff = Ops.sqr(_diff);
                variance = Ops.mean(_sqrDiff, 0, 1, 2);
            } else {
                mean = new Tensor(averages.runningMean, Tensor.GradientMode.NONE);
                variance = new Tensor(averages.runningVariance, Tensor.GradientMode.NONE);
                _diff = Ops.sub(input, Ops.reshape(mean, 1, 1, 1, channels));
            }
            // X_hat = (X - mean.reshape((1, C, 1, 1))) * 1.0 / nd.sqrt(variance.reshape((1, C, 1, 1)) + eps)
            Tensor _sqrVarianceWithEpsilon = Ops.add(variance, Ops.EPSILON);
            Tensor _sqrtVariance = Ops.sqrt(_sqrVarianceWithEpsilon);
            Tensor _normalized = Ops.div(_diff, _sqrtVariance);
            //        out = gamma.reshape((1, C, 1, 1)) * X_hat + beta.reshape((1, C, 1, 1))
            Tensor _scaled = Ops.mul(_normalized, gamma);
            Tensor scaledAndShifted = Ops.add(_scaled, beta);

            return new BatchNormResult(scaledAndShifted, mean.getVals(), variance.getVals());
        } else {
            Tensor mean;
            Tensor _diff;
            Tensor variance;

            if (runMode == Ops.RunMode.TRAINING) {
                mean = Ops.mean(input, 0);
                _diff = Ops.sub(input, mean);
                Tensor _sqrDiff = Ops.sqr(_diff);
                variance = Ops.mean(_sqrDiff, 0);
            } else {
                mean = new Tensor(averages.runningMean, Tensor.GradientMode.NONE);
                variance = new Tensor(averages.runningVariance, Tensor.GradientMode.NONE);
                _diff = Ops.sub(input, mean);
            }
            Tensor _sqrVarianceWithEpsilon = Ops.add(variance, Ops.EPSILON);
            Tensor _sqrtVariance = Ops.sqrt(_sqrVarianceWithEpsilon);
            Tensor _normalized = Ops.div(_diff, _sqrtVariance);
            Tensor _scaled = Ops.mul(_normalized, gamma);
            Tensor scaledAndShifted = Ops.add(_scaled, beta);

            return new BatchNormResult(scaledAndShifted, mean.getVals(), variance.getVals());
        }
    }

    public static class BatchNormRunningAverages implements DisposableContainer {
        public final NDArray runningMean;
        public final NDArray runningVariance;

        public BatchNormRunningAverages() {
            this(null, null);
        }

        public BatchNormRunningAverages(NDArray runningMean, NDArray runningVariance) {
            this.runningMean = runningMean;
            this.runningVariance = runningVariance;
        }

        public BatchNormRunningAverages updateWith(BatchNormResult result, double momentum) {
            NDArray newVariance = update(this.runningVariance, result.variance, momentum);
            NDArray newMean = update(this.runningMean, result.mean, momentum);

            return new BatchNormRunningAverages(newMean, newVariance);
        }

        private static NDArray update(NDArray old, NDArray current, double momentum) {
            if (old == null) {
                return current;
            } else {
                NDArray newAmount = current.mul(1.0 - momentum);
                NDArray keptFromOld = old.mul(momentum);
                return keptFromOld.add(newAmount);
            }
        }

        @Override
        public String toString() {
            return "Mean: " + runningMean + "\n" +
                    "Variance: " + runningVariance;
        }

        @Override
        public List<NDArray> getDisposables() {
            return asList(runningMean, runningVariance);
        }

        public void registerForDisposal() {
            DisposalRegister.registerForDisposal(this.runningMean, this.runningVariance);
        }
    }

    public static class BatchNormResult implements DisposableContainer<NDArray> {
        public final Tensor output;
        public final NDArray mean;
        public final NDArray variance;

        BatchNormResult(Tensor output, NDArray mean, NDArray variance) {
            this.output = output;
            this.mean = mean;
            this.variance = variance;
        }

        @Override
        public List<NDArray> getDisposables() {
            return asList(output.getVals(), output.getGradient(), mean, variance);
        }
    }
}
