package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.array.JavaArray;
import com.codeberry.tadlib.array.TMutableArray;
import com.codeberry.tadlib.nn.loss.SoftmaxCrossEntropyLoss;
import com.codeberry.tadlib.util.MultiThreadingSupport.TaskRange;

import java.util.List;
import java.util.Random;

import static com.codeberry.tadlib.array.JavaArray.DimKeepRemove.KEEP_DIM;
import static com.codeberry.tadlib.array.JavaArray.DimKeepRemove.REMOVE_DIM;
import static com.codeberry.tadlib.array.TArrayFactory.*;
import static com.codeberry.tadlib.tensor.ParentLink.parentLink;
import static com.codeberry.tadlib.util.MultiThreadingSupport.TaskRange.taskRange;
import static com.codeberry.tadlib.util.MultiThreadingSupport.multiThreadingSupportRun;
import static java.lang.Boolean.TRUE;
import static java.util.Arrays.*;
import static java.util.Collections.singletonList;
import static java.util.stream.Collectors.toList;

public abstract class Ops {
    public static final double EPSILON = 0.000001;

    public static Tensor matmul(Tensor a, Tensor b) {
        JavaArray y = a.vals.matmul(b.vals);

        GradFunc gF_a = grad -> grad.matmul(b.vals.transpose());
        GradFunc gF_b = grad -> a.vals.transpose().matmul(grad);

        return new Tensor(y,
                asList(parentLink(a, gF_a), parentLink(b, gF_b)));
    }

    public static Tensor negate(Tensor a) {
        JavaArray y = a.vals.negate();

        GradFunc gF = JavaArray::negate;

        return new Tensor(y,
                singletonList(parentLink(a, gF)));
    }

    public static Tensor sub(Tensor a, Tensor b) {
        Tensor negated = negate(b);
        return add(a, negated);
    }

    public static Tensor add(Tensor... tensors) {
        JavaArray y = stream(tensors)
                .map(t -> t.vals)
                .reduce((val1, val2) -> val1.add(val2))
                .orElse(JavaArray.ZERO);

        List<ParentLink> parents = stream(tensors)
                .map(t -> parentLink(t, funcGradientAdd(t)))
                .collect(toList());

        return new Tensor(y, parents);
    }

    public static Tensor add(Tensor a, Tensor b) {
        JavaArray y = a.vals.add(b.vals);

        GradFunc gF_a = funcGradientAdd(a);
        GradFunc gF_b = funcGradientAdd(b);

        return new Tensor(y,
                asList(parentLink(a, gF_a), parentLink(b, gF_b)));
    }

    public static Tensor add(Tensor a, double constant) {
        JavaArray y = a.vals.add(constant);

        GradFunc gF_a = funcGradientAdd(a);

        return new Tensor(y, singletonList(parentLink(a, gF_a)));
    }

    private static GradFunc funcGradientAdd(Tensor tensor) {
        return grad -> aggregateBroadcastedDims(tensor, grad);
    }

    public static Tensor sqr(Tensor a) {
        JavaArray y = a.vals.sqr();

        GradFunc gF = grad -> {
            grad = grad.mul(a.vals).mul(2.0);

            return aggregateBroadcastedDims(a, grad);
        };

        return new Tensor(y, singletonList(parentLink(a, gF)));
    }

    public static Tensor sqrt(Tensor a) {
        JavaArray y = a.vals.sqrt();

        GradFunc gF = grad -> {
            grad = grad.mul(a.vals.pow(-0.5).mul(0.5));

            return aggregateBroadcastedDims(a, grad);
        };

        return new Tensor(y, singletonList(parentLink(a, gF)));
    }

    public static Tensor div(Tensor a, Tensor b) {
        JavaArray y = a.vals.div(b.vals);

        GradFunc gF_a = grad -> {
            JavaArray agged = aggregateBroadcastedDims(a, grad);

            return agged.div(b.vals);
        };
        GradFunc gF_b = grad -> {
            //  calced_divisor_grad = tf.reduce_sum(fake_grad * (-dividend / (divisor**2)), axis=(0,1))
            JavaArray negateA = a.vals.negate();
            JavaArray rawGrad = grad.mul(negateA);
            JavaArray sqrB = b.vals.sqr();
            JavaArray gradForB = rawGrad.div(sqrB);

            return aggregateBroadcastedDims(b, gradForB);
        };

        return new Tensor(y,
                asList(parentLink(a, gF_a), parentLink(b, gF_b)));
    }

    public static Tensor mul(Tensor a, Tensor b) {
        JavaArray y = a.vals.mul(b.vals);

        GradFunc gF_a = funcGradientMul(a, b);
        GradFunc gF_b = funcGradientMul(b, a);

        return new Tensor(y,
                asList(parentLink(a, gF_a), parentLink(b, gF_b)));
    }

    private static GradFunc funcGradientMul(Tensor self, Tensor other) {
        return grad -> {
            grad = grad.mul(other.vals);

            return aggregateBroadcastedDims(self, grad);
        };
    }

    private static JavaArray aggregateBroadcastedDims(Tensor self, JavaArray grad) {
        int missingDims = grad.shape.dimCount - self.vals.shape.dimCount;
        if (missingDims >= 1) {
            grad = grad.sumFirstDims(missingDims, REMOVE_DIM);
        }

        if (self.vals.shape.hasSingleDims()) {
            Boolean[] dimensionsToSum = self.vals.shape.forEachDim(dim -> dim == 1, Boolean[]::new);
            grad = grad.sumDims(dimensionsToSum, KEEP_DIM);
        }
        return grad;
    }

    public static Tensor sum(Tensor a) {
        JavaArray y = a.vals.sum();

        GradFunc gF = grad -> fillLike(a.vals.shape, grad);

        return new Tensor(y,
                singletonList(parentLink(a, gF)));
    }

    public static Tensor conv2d(Tensor input, Tensor filter) {
        JavaArray y = input.vals.conv2d(filter.vals);

        GradFunc gF_Input = grad -> {
            JavaArray transposed = filter.vals.transpose(0, 1, 3, 2);
            JavaArray fT = transposed.normalOrderedCopy();
            JavaArray fRotated = fT.rot180();

            int offsetY = (filter.vals.shape.at(0) % 2 == 0) ? -1 : 0;
            int offsetX = (filter.vals.shape.at(1) % 2 == 0) ? -1 : 0;
            return grad.conv2d(fRotated, offsetY, offsetX, JavaArray.Debug.NONE);
        };
        GradFunc gF_Filter = grad -> calcFilterGradient(grad, input.vals, filter.vals);

        return new Tensor(y,
                asList(parentLink(input, gF_Input),
                        parentLink(filter, gF_Filter)));
    }

    private static JavaArray calcFilterGradient(JavaArray grad, JavaArray input, JavaArray filter) {
        Shape filterShape = filter.shape;
        int[] dims = new int[filterShape.dimCount + 1];
        System.arraycopy(filterShape.toDimArray(), 0, dims, 1, filterShape.dimCount);
        dims[0] = input.shape.at(0);
        Shape tgtShape = new Shape(dims);

        JavaArray gradPerInputExample = multiThreadingSupportRun(taskRange(0, grad.shape.at(0)),
                range -> accumulateFilterGradientAtFirstDim(range, grad, input, tgtShape),
                (left, right) -> left.add(right));

        return gradPerInputExample.sumFirstDims(1, REMOVE_DIM);
    }

    private static JavaArray accumulateFilterGradientAtFirstDim(TaskRange range, JavaArray grad, JavaArray input, Shape tgtShape) {
        TMutableArray tgtGrad = new TMutableArray(new double[tgtShape.size], tgtShape);

        int[] gradIndices = grad.shape.newIndexArray();
        int[] inIndices = input.shape.newIndexArray();
        int[] tgtIndices = tgtShape.newIndexArray();

        for (int i = range.start; i < range.end; i++) {
            gradIndices[0] = i;
            inIndices[0] = i;
            tgtIndices[0] = i;
            accumulateFilterGradient(grad, gradIndices, 1, input, inIndices, tgtGrad, tgtIndices);
        }

        return tgtGrad.migrateToImmutable();
    }

    private static void accumulateFilterGradient(JavaArray grad, int[] gradIndices, int dim,
                                                 JavaArray input, int[] inIndices,
                                                 TMutableArray tgtGrad, int[] tgtIndices) {
        if (gradIndices.length - dim == 3) {
            int filterH = tgtGrad.shape.at(1);
            int filterW = tgtGrad.shape.at(2);
            int inputChannels = input.shape.at(-1);
            int outChannels = tgtGrad.shape.at(-1);
            int tgtDims = tgtIndices.length;
            for (int inIdx = 0; inIdx < inputChannels; inIdx++) {
                for (int outIdx = 0; outIdx < outChannels; outIdx++) {
                    for (int y = 0; y < filterH; y++) {
                        for (int x = 0; x < filterW; x++) {
                            double g = sumFilterGradAt(filterH, filterW,
                                    grad, gradIndices,
                                    input, inIndices,
                                    inIdx, outIdx,
                                    y, x);

                            tgtIndices[tgtDims - 4] = y;
                            tgtIndices[tgtDims - 3] = x;
                            tgtIndices[tgtDims - 2] = inIdx;
                            tgtIndices[tgtDims - 1] = outIdx;
                            tgtGrad.setAt(tgtIndices, g);
                        }
                    }
                }
            }
        } else {
            int len = grad.shape.at(dim);
            for (int i = 0; i < len; i++) {
                gradIndices[dim] = i;
                inIndices[dim] = i;
                tgtIndices[dim] = i;
                accumulateFilterGradient(grad, gradIndices, dim + 1, input, inIndices, tgtGrad, tgtIndices);
            }
        }
    }

    private static double sumFilterGradAt(int filterH, int filterW,
                                          JavaArray grad, int[] gradIndices,
                                          JavaArray input, int[] inIndices,
                                          int inIdx, int outIdx,
                                          int offsetGradY, int offsetGradX) {
        int inputH = input.shape.at(-3);
        int inputW = input.shape.at(-2);
        int offsetY = (filterH - 1) / 2;
        int offsetX = (filterW - 1) / 2;
        int len = gradIndices.length;
        gradIndices[len - 1] = outIdx;
        inIndices[len - 1] = inIdx;
        double g = 0;
        for (int y = 0; y < inputH; y++) {
            for (int x = 0; x < inputW; x++) {
                inIndices[len - 3] = y;
                inIndices[len - 2] = x;
                double inputVal = input.dataAt(inIndices);

                gradIndices[len - 3] = y - offsetGradY + offsetY;
                gradIndices[len - 2] = x - offsetGradX + offsetX;
                double gradVal = grad.dataAt(gradIndices);

                g += inputVal * gradVal;
            }
        }
        return g;
    }

    public static Tensor maxpool2d(Tensor input, int size) {
        Shape outShape = getMaxPool2dOutputSize(input.vals.shape, size);

        TMutableArray tgt = new TMutableArray(new double[outShape.size], outShape);

        int[] inputIndices = input.vals.shape.newIndexArray();
        int[] tgtIndices = outShape.newIndexArray();
        Shape maxIndexShape = createMax2dIndexShape(outShape);
        int[] tmpMaxIndices = maxIndexShape.newIndexArray();
        int[] maxIndexData = new int[maxIndexShape.size];

        fillMax2d(input.vals, inputIndices, size, tgt, tgtIndices,
                maxIndexShape, maxIndexData, tmpMaxIndices,
                0);

        GradFunc gF = grad -> distribute2dMaxGrad(grad, input.vals.shape, maxIndexShape, maxIndexData);

        return new Tensor(tgt.migrateToImmutable(), singletonList(parentLink(input, gF)));
    }

    public static Shape getMaxPool2dOutputSize(Shape inputShape, int size) {
        int[] dims = inputShape.toDimArray();
        int len = dims.length;
        int newH = (dims[len - 3] + size - 1) / size;
        int newW = (dims[len - 2] + size - 1) / size;
        dims[len - 3] = newH;
        dims[len - 2] = newW;

        return new Shape(dims);
    }

    private static JavaArray distribute2dMaxGrad(JavaArray grad, Shape inputShape, Shape maxIndexShape, int[] maxIndexData) {
        TMutableArray outputGrad = new TMutableArray(inputShape);

        int[] tmpOutputGradIndices = outputGrad.shape.newIndexArray();
        int[] tmpGradIndices = grad.shape.newIndexArray();
        int[] tmpMaxIndices = maxIndexShape.newIndexArray();
        fillMax2dGradInto(outputGrad, maxIndexShape, maxIndexData, grad, 0,
                tmpOutputGradIndices, tmpGradIndices, tmpMaxIndices);

        return outputGrad.migrateToImmutable();
    }

    private static void fillMax2dGradInto(TMutableArray outputGrad, Shape maxIndexShape, int[] maxIndexData, JavaArray grad, int dim,
                                          int[] tmpOutputGradIndices, int[] tmpGradIndices, int[] tmpMaxIndices) {
        if (maxIndexShape.dimCount - dim == 4) {
            int maxILen = tmpMaxIndices.length;
            int gradLen = tmpGradIndices.length;
            int outputLen = tmpOutputGradIndices.length;
            int h = grad.shape.at(-3);
            int w = grad.shape.at(-2);
            int channels = grad.shape.at(-1);
            for (int y = 0; y < h; y++) {
                tmpGradIndices[gradLen - 3] = y;
                tmpMaxIndices[maxILen - 4] = y;
                for (int x = 0; x < w; x++) {
                    tmpGradIndices[gradLen - 2] = x;
                    tmpMaxIndices[maxILen - 3] = x;
                    for (int c = 0; c < channels; c++) {
                        tmpGradIndices[gradLen - 1] = c;
                        tmpMaxIndices[maxILen - 2] = c;

                        // y
                        tmpMaxIndices[maxILen - 1] = 0;
                        int indexYOffset = maxIndexShape.calcDataIndex(tmpMaxIndices);
                        int yInInput = maxIndexData[indexYOffset];
                        // x
                        tmpMaxIndices[maxILen - 1] = 1;
                        int indexXOffset = maxIndexShape.calcDataIndex(tmpMaxIndices);
                        int xInInput = maxIndexData[indexXOffset];

                        double gVal = grad.dataAt(tmpGradIndices);

                        tmpOutputGradIndices[outputLen - 3] = yInInput;
                        tmpOutputGradIndices[outputLen - 2] = xInInput;
                        tmpOutputGradIndices[outputLen - 1] = c;
                        outputGrad.setAt(tmpOutputGradIndices, gVal);
                    }
                }
            }
        } else {
            int len = maxIndexShape.at(dim);
            for (int i = 0; i < len; i++) {
                tmpOutputGradIndices[dim] = i;
                tmpGradIndices[dim] = i;
                tmpMaxIndices[dim] = i;
                fillMax2dGradInto(outputGrad, maxIndexShape, maxIndexData, grad, dim + 1,
                        tmpOutputGradIndices, tmpGradIndices, tmpMaxIndices);
            }
        }
    }

    private static Shape createMax2dIndexShape(Shape outShape) {
        int[] idxDims = copyOf(outShape.toDimArray(), outShape.dimCount + 1);
        // for (y,x) pair to log the original location of the value in
        // the input matrix.
        idxDims[outShape.dimCount] = 2;
        return new Shape(idxDims);
    }

    private static void fillMax2d(JavaArray input, int[] inputIndices, int size,
                                  TMutableArray tgt, int[] tgtIndices,
                                  Shape maxIndexShape, int[] maxIndexData, int[] tmpMaxIndices,
                                  int dim) {
        if (inputIndices.length - dim == 3) {
            int len = tgtIndices.length;
            int h = tgt.shape.at(-3);
            int w = tgt.shape.at(-2);
            int channels = tgt.shape.at(-1);
            for (int y = 0; y < h; y++) {
                tgtIndices[len - 3] = y;
                tmpMaxIndices[len - 3] = y;
                for (int x = 0; x < w; x++) {
                    tgtIndices[len - 2] = x;
                    tmpMaxIndices[len - 2] = x;
                    for (int c = 0; c < channels; c++) {
                        tgtIndices[len - 1] = c;
                        tmpMaxIndices[len - 1] = c;

                        double maxVal = getMax2dVal(input, inputIndices,
                                y * size, x * size, c, size,
                                maxIndexShape, maxIndexData, tmpMaxIndices);

                        tgt.setAt(tgtIndices, maxVal);
                    }
                }
            }
        } else {
            int len = input.shape.at(dim);
            for (int i = 0; i < len; i++) {
                inputIndices[dim] = i;
                tgtIndices[dim] = i;
                tmpMaxIndices[dim] = i;

                fillMax2d(input, inputIndices, size, tgt, tgtIndices,
                        maxIndexShape, maxIndexData, tmpMaxIndices,
                        dim + 1);
            }
        }
    }

    private static double getMax2dVal(JavaArray input, int[] inputIndices,
                                      int yInputOffset, int xInputOffset, int c,
                                      int size,
                                      Shape maxIndexShape, int[] maxIndexData, int[] tmpMaxIndices) {
        double max = Double.NEGATIVE_INFINITY;
        int len = inputIndices.length;
        inputIndices[len - 1] = c;
        int maxY = -1;
        int maxX = -1;

        int inputH = input.shape.at(-3);
        int inputW = input.shape.at(-2);

        for (int y = 0; y < size; y++) {
            int inY = y + yInputOffset;

            if (inY < inputH) {
                inputIndices[len - 3] = inY;
                for (int x = 0; x < size; x++) {
                    int inX = x + xInputOffset;
                    if (inX < inputW) {
                        inputIndices[len - 2] = inX;
                        double inVal = input.dataAt(inputIndices);
                        if (inVal > max) {
                            maxY = inY;
                            maxX = inX;
                            max = inVal;
                        }
                    }
                }
            }
        }

        // y
        tmpMaxIndices[maxIndexShape.dimCount - 1] = 0;
        int indexYOffset = maxIndexShape.calcDataIndex(tmpMaxIndices);
        maxIndexData[indexYOffset] = maxY;
        // x
        tmpMaxIndices[maxIndexShape.dimCount - 1] = 1;
        int indexXOffset = maxIndexShape.calcDataIndex(tmpMaxIndices);
        maxIndexData[indexXOffset] = maxX;

        return max;
    }

    public static Tensor flatten(Tensor input) {
        Shape inputShape = input.vals.shape;
        int size = calcFlattenExampleSize(inputShape);

        GradFunc gF = grad -> grad.reshape(inputShape);

        return new Tensor(input.vals.reshape(inputShape.at(0), size),
                singletonList(parentLink(input, gF)));
    }

    public static int calcFlattenExampleSize(Shape inputShape) {
        int size = 1;
        for (int i = 1; i < inputShape.dimCount; i++) {
            size *= inputShape.at(i);
        }
        return size;
    }

    public static Tensor reshape(Tensor input, int... dims) {
        Shape inputShape = input.vals.shape;

        GradFunc gF = grad -> grad.reshape(inputShape);

        return new Tensor(input.vals.reshape(dims),
                singletonList(parentLink(input, gF)));
    }

    public static Tensor relu(Tensor input) {
        JavaArray copy = input.vals.normalOrderedCopy();
        double[] data = copy.getInternalData();
        double[] gradMaskData = new double[data.length];

        for (int i = 0; i < data.length; i++) {
            if (data[i] <= 0) {
                data[i] = 0;
            } else {
                gradMaskData[i] = 1;
            }
        }

        Shape copyShape = copy.shape;
        GradFunc gF = grad -> grad.mul(new JavaArray(gradMaskData, copyShape.copy()));

        return new Tensor(copy, singletonList(parentLink(input, gF)));
    }

    public static Tensor leakyRelu(Tensor input, double leakyScale) {
        JavaArray copy = input.vals.normalOrderedCopy();
        double[] data = copy.getInternalData();
        double[] gradMaskData = new double[data.length];

        for (int i = 0; i < data.length; i++) {
            if (data[i] <= 0) {
                data[i] *= leakyScale;
                gradMaskData[i] = leakyScale;
            } else {
                gradMaskData[i] = 1;
            }
        }

        Shape copyShape = copy.shape;
        GradFunc gF = grad -> grad.mul(new JavaArray(gradMaskData, copyShape.copy()));

        return new Tensor(copy, singletonList(parentLink(input, gF)));
    }

    public static Tensor sumSoftmaxCrossEntropy(Tensor labelsOneHot, Tensor prediction) {
        JavaArray softmax = prediction.vals.softmax();

        double cost = SoftmaxCrossEntropyLoss.sumSoftmaxCrossEntropy(softmax, softmax.shape.newIndexArray(),
                labelsOneHot.vals, 0);

        GradFunc gF = grad -> {
            TMutableArray tgt = TMutableArray.copyOf(softmax);
            toSoftmaxGradient(tgt, softmax, softmax.shape.newIndexArray(),
                    labelsOneHot.vals, 0);
            return tgt.migrateToImmutable().mul(grad);
        };

        return new Tensor(new JavaArray(cost), singletonList(parentLink(prediction, gF)));
    }

    private static void toSoftmaxGradient(TMutableArray tgt, JavaArray predicted, int[] indices, JavaArray labelsOneHot, int dim) {
        int len = predicted.shape.at(dim);
        if (indices.length - dim == 1) {
            int maxIndex = -1;
            double max = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < len; i++) {
                indices[dim] = i;
                double tgtVal = labelsOneHot.dataAt(indices);
                if (tgtVal > max) {
                    max = tgtVal;
                    maxIndex = i;
                }
            }

            indices[dim] = maxIndex;
            double pred = predicted.dataAt(indices);
            tgt.setAt(indices, pred - 1);
        } else {
            for (int i = 0; i < len; i++) {
                indices[dim] = i;
                toSoftmaxGradient(tgt, predicted, indices, labelsOneHot, dim + 1);
            }
        }
    }

    public static Tensor dropout(Tensor input, Random rnd, double dropoutKeep, RunMode runMode) {
        if (runMode == RunMode.TRAINING) {
            double gradValue = 1.0 / dropoutKeep;
            JavaArray output = input.vals.normalOrderedCopy();
            double[] data = output.getInternalData();
            double[] gradMaskData = new double[data.length];
            int[] dims = output.shape.toDimArray();
            for (int i = 0; i < data.length; i++) {
                if (rnd.nextDouble() >= dropoutKeep) {
                    data[i] = 0;
                } else {
                    data[i] /= dropoutKeep;
                    gradMaskData[i] = gradValue;
                }
            }

            GradFunc gF = grad -> grad.mul(new JavaArray(gradMaskData, Shape.of(dims)));

            return new Tensor(output, singletonList(parentLink(input, gF)));
        }
        return input;
    }

    public static Tensor mean(Tensor input, int... axis) {
        Shape inputShape = input.vals.shape;
        Boolean[] dimsToCollapse = inputShape.newCollapseArray();
        for (int axi : axis) {
            dimsToCollapse[axi] = TRUE;
        }
        int elementsSummed = 1;
        for (int i = 0; i < inputShape.dimCount; i++) {
            if (dimsToCollapse[i]) {
                elementsSummed *= inputShape.at(i);
            }
        }
        JavaArray sum = input.vals.sumDims(dimsToCollapse, REMOVE_DIM);
        JavaArray mean = sum.div(elementsSummed);

        int[] broadcastDims = inputShape.toDimArray();
        for (int d = 0; d < dimsToCollapse.length; d++) {
            if (dimsToCollapse[d]) {
                broadcastDims[d] = 1;
            }
        }

        int finalElementsSummed = elementsSummed;
        GradFunc gF = grad -> {
            JavaArray broadcastCompatGrad = grad.reshape(broadcastDims);
            JavaArray onesInInputShape = ones(inputShape);
            JavaArray gradInInputShape = onesInInputShape.mul(broadcastCompatGrad);

            return gradInInputShape.div(finalElementsSummed);
        };

        return new Tensor(mean, singletonList(parentLink(input, gF)));
    }

    public enum RunMode {
        TRAINING, INFERENCE
    }
}
