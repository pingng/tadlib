package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.array.TArray;

public abstract class OpsExtended {
    public static BatchNormResult batchNorm(Tensor input, Tensor beta, Tensor gamma, BatchNormRunningAverages averages, Ops.RunMode runMode) {
        Shape shape = input.vals.shape;
        if (shape.dimCount != 2 && shape.dimCount != 4) {
            throw new IllegalArgumentException("Valid dims are 2 an 4");
        }

        if (shape.dimCount == 4) {
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

            return new BatchNormResult(scaledAndShifted, mean.vals, variance.vals);
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

            return new BatchNormResult(scaledAndShifted, mean.vals, variance.vals);
        }
    }

    public static class BatchNormRunningAverages {
        final TArray runningMean;
        final TArray runningVariance;

        public BatchNormRunningAverages() {
            this(null, null);
        }

        public BatchNormRunningAverages(TArray runningMean, TArray runningVariance) {
            this.runningMean = runningMean;
            this.runningVariance = runningVariance;
        }

        public BatchNormRunningAverages updateWith(BatchNormResult result, double momentum) {
            TArray newVariance = update(this.runningVariance, result.variance, momentum);
            TArray newMean = update(this.runningMean, result.mean, momentum);

            return new BatchNormRunningAverages(newMean, newVariance);
        }

        private static TArray update(TArray old, TArray current, double momentum) {
            if (old == null) {
                return current;
            } else {
                TArray newAmount = current.mul(1.0 - momentum);
                TArray keptFromOld = old.mul(momentum);
                return keptFromOld.add(newAmount);
            }
        }

        @Override
        public String toString() {
            return "Mean: " + runningMean + "\n" +
                    "Variance: " + runningVariance;
        }
    }

    public static class BatchNormResult {
        public final Tensor output;
        public final TArray mean;
        public final TArray variance;

        BatchNormResult(Tensor output, TArray mean, TArray variance) {
            this.output = output;
            this.mean = mean;
            this.variance = variance;
        }
    }
}
