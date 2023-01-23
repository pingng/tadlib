package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.memorymanagement.DisposalRegister;
import com.codeberry.tadlib.provider.java.NDArray;

import java.util.List;

import static com.codeberry.tadlib.tensor.Tensor.constant;
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
        NDArray ndArray = input.val();
        Shape shape = ndArray.shape;
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
                BatchNormRunningAverages.Data r = averages.running;
                mean = constant(r.mean);
                variance = constant(r.variance);
                _diff = Ops.sub(input, Ops.reshape(mean, 1, 1, 1, channels));
            }
            // X_hat = (X - mean.reshape((1, C, 1, 1))) * 1.0 / nd.sqrt(variance.reshape((1, C, 1, 1)) + eps)
            Tensor _sqrVarianceWithEpsilon = Ops.add(variance, Ops.EPSILON);
            Tensor _sqrtVariance = Ops.sqrt(_sqrVarianceWithEpsilon);
            Tensor _normalized = Ops.div(_diff, _sqrtVariance);
            //        out = gamma.reshape((1, C, 1, 1)) * X_hat + beta.reshape((1, C, 1, 1))
            Tensor _scaled = Ops.mul(_normalized, gamma);
            Tensor scaledAndShifted = Ops.add(_scaled, beta);

            return new BatchNormResult(scaledAndShifted, mean.val(), variance.val());
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
                BatchNormRunningAverages.Data r = averages.running;
                mean = constant(r.mean);
                variance = constant(r.variance);
                _diff = Ops.sub(input, mean);
            }
            Tensor _sqrVarianceWithEpsilon = Ops.add(variance, Ops.EPSILON);
            Tensor _sqrtVariance = Ops.sqrt(_sqrVarianceWithEpsilon);
            Tensor _normalized = Ops.div(_diff, _sqrtVariance);
            Tensor _scaled = Ops.mul(_normalized, gamma);
            Tensor scaledAndShifted = Ops.add(_scaled, beta);

            return new BatchNormResult(scaledAndShifted, mean.val(), variance.val());
        }
    }

    public static class BatchNormRunningAverages {
        volatile Data running;

        public BatchNormRunningAverages() {
            this(null, null);
        }

        public BatchNormRunningAverages(NDArray runningMean, NDArray runningVariance) {
            this.running = new Data(runningMean, runningVariance);
        }

        public void updateWith(BatchNormResult result, double momentum) {
            Data old = this.running;

            NDArray newVariance = update(old.variance, result.variance, momentum);
            NDArray newMean = update(old.mean, result.mean, momentum);

            this.running = new Data(newMean, newVariance);

            old.registerForDisposal();
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
            Data r = this.running;
            return "Mean: " + r.mean + "\n" +
                    "Variance: " + r.variance;
        }

        public List<NDArray> getKeepInMemoryDisposables() {
            Data r = this.running;
            return asList(r.mean, r.variance);
        }

        public static class Data {
            public final NDArray mean;
            public final NDArray variance;

            Data(NDArray mean, NDArray variance) {
                this.mean = mean;
                this.variance = variance;
            }

            void registerForDisposal() {
                DisposalRegister.registerForDisposal(this.mean, this.variance);
            }
        }
    }

    public static class BatchNormResult {
        public final Tensor output;
        public final NDArray mean;
        public final NDArray variance;

        BatchNormResult(Tensor output, NDArray mean, NDArray variance) {
            this.output = output;
            this.mean = mean;
            this.variance = variance;
        }
    }
}
