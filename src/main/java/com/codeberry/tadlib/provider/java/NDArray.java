package com.codeberry.tadlib.provider.java;

import com.codeberry.tadlib.array.Comparison;
import com.codeberry.tadlib.array.exception.DimensionMissing;
import com.codeberry.tadlib.array.util.FlatToMultiDimArrayConverter;
import com.codeberry.tadlib.array.util.SoftmaxUtils;
import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.util.MultiThreadingSupport;
import com.codeberry.tadlib.util.StringUtils;
import com.codeberry.tadlib.util.memory.DisposalRegister;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.IntFunction;

import static com.codeberry.tadlib.array.util.DimensionUtils.*;
import static com.codeberry.tadlib.provider.java.NDArray.DimKeepRemove.REMOVE_DIM;
import static com.codeberry.tadlib.util.MultiThreadingSupport.TaskRange.taskRange;
import static com.codeberry.tadlib.util.MultiThreadingSupport.multiThreadingSupportRun;
import static java.lang.Math.*;
import static java.util.Arrays.*;

public class NDArray implements DisposalRegister.Disposable {
    public final double[] data;
    public final Shape shape;

    public NDArray(double val) {
        this(new double[]{val}, Shape.zeroDim);
    }

    public NDArray(double[] data) {
        this(data, new Shape(data.length));
    }

    public NDArray(Shape shape) {
        this(new double[shape.size], shape);
    }

    public NDArray(double[] data, Shape shape) {
        this.data = data;
        this.shape = shape;
    }

    public static NDArray distribute2dMaxGrad(NDArray grad, Shape inputShape, Shape maxIndexShape, int[] maxIndexData) {
        TMutableArray outputGrad = new TMutableArray(inputShape);

        int[] tmpOutputGradIndices = outputGrad.shape.newIndexArray();
        int[] tmpGradIndices = grad.shape.newIndexArray();
        int[] tmpMaxIndices = maxIndexShape.newIndexArray();
        fillMax2dGradInto(outputGrad, maxIndexShape, maxIndexData, grad, 0,
                tmpOutputGradIndices, tmpGradIndices, tmpMaxIndices);

        return outputGrad.migrateToImmutable();
    }

    public static void validateConv2dShapes(Shape inputShape, Shape filterShape) {
        if (inputShape.dimCount < 4) {
            throw new RuntimeException("input must have 4+ dims");
        }
        if (filterShape.dimCount != 4) {
            throw new RuntimeException("filter must have dims [h,w,in,out]");
        }
    }

    public void set(NDArray x) {
        set(x.data);
    }

    public final boolean setIfDifferent(NDArray x) {
        if (!Arrays.equals(data, x.data)) {
            set(x);
            return true;
        }
        return false;
    }

    public void set(double[] x) {
        System.arraycopy(x, 0, data, 0, data.length);
    }

    private static void fillMax2dGradInto(TMutableArray outputGrad, Shape maxIndexShape, int[] maxIndexData, NDArray grad, int dim,
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

    private static void toSoftmaxGradientOLD(TMutableArray tgt, NDArray predicted, int[] indices, NDArray labelsOneHot, int dim) {
        int len = predicted.shape.at(dim);
        if (indices.length - dim == 1) {
            // --- Find MAX index in last dim ---
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

            // --- Only change value of MAX INDEX ---
            indices[dim] = maxIndex;
            double pred = predicted.dataAt(indices);
            tgt.setAt(indices, pred - 1);
        } else {
            for (int i = 0; i < len; i++) {
                indices[dim] = i;
                toSoftmaxGradientOLD(tgt, predicted, indices, labelsOneHot, dim + 1);
            }
        }
    }

    public NDArray calcConv2dFilterGradient(NDArray input, NDArray filter) {
        Shape filterShape = filter.shape;
        int[] dims = new int[filterShape.dimCount + 1];
        System.arraycopy(filterShape.toDimArray(), 0, dims, 1, filterShape.dimCount);
        dims[0] = input.shape.at(0);
        Shape tgtShape = new Shape(dims);

        NDArray gradPerInputExample = multiThreadingSupportRun(taskRange(0, shape.at(0)),
                range -> accumulateFilterGradientAtFirstDim(range, input, tgtShape),
                (left, right) -> left.add(right));

        return gradPerInputExample.sumFirstDims(1, REMOVE_DIM);
    }

    private NDArray accumulateFilterGradientAtFirstDim(MultiThreadingSupport.TaskRange range, NDArray input, Shape tgtShape) {
        TMutableArray tgtGrad = new TMutableArray(new double[tgtShape.size], tgtShape);

        int[] gradIndices = this.shape.newIndexArray();
        int[] inIndices = input.shape.newIndexArray();
        int[] tgtIndices = tgtShape.newIndexArray();

        for (int i = range.start; i < range.end; i++) {
            gradIndices[0] = i;
            inIndices[0] = i;
            tgtIndices[0] = i;
            accumulateFilterGradient(gradIndices, 1, input, inIndices, tgtGrad, tgtIndices);
        }

        return tgtGrad.migrateToImmutable();
    }

    private void accumulateFilterGradient(int[] gradIndices, int dim,
                                          NDArray input, int[] inIndices,
                                          TMutableArray tgtGrad, int[] tgtIndices) {
        if (gradIndices.length - dim == 3) {
            int filterH = tgtGrad.shape.at(1);
            int filterW = tgtGrad.shape.at(2);
            int inputChannels = input.shape.at(-1);
            int outChannels = tgtGrad.shape.at(-1);
            int tgtDims = tgtIndices.length;
//            System.out.println("--- Example " + gradIndices[0] + " ---");
            for (int inIdx = 0; inIdx < inputChannels; inIdx++) {
                for (int outIdx = 0; outIdx < outChannels; outIdx++) {
                    for (int y = 0; y < filterH; y++) {
                        for (int x = 0; x < filterW; x++) {
//                            System.out.println("- filterGrad["+y+"][" + x+"][" + inIdx+"][" + outIdx+"] -");
                            double g = sumFilterGradAt(filterH, filterW,
                                    gradIndices,
                                    input, inIndices,
                                    inIdx, outIdx,
                                    y, x);
//                            System.out.println("= " + g);
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
            int len = shape.at(dim);
            for (int i = 0; i < len; i++) {
                gradIndices[dim] = i;
                inIndices[dim] = i;
                tgtIndices[dim] = i;
                accumulateFilterGradient(gradIndices, dim + 1, input, inIndices, tgtGrad, tgtIndices);
            }
        }
    }

    private double sumFilterGradAt(int filterH, int filterW,
                                   int[] gradIndices,
                                   NDArray input, int[] inIndices,
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
//                System.out.print("input...[" + inIndices[len - 3] + "][" + inIndices[len - 2] + "][" + inIndices[len - 1] + "]" );

                gradIndices[len - 3] = y - offsetGradY + offsetY;
                gradIndices[len - 2] = x - offsetGradX + offsetX;
                double gradVal = dataAt(gradIndices);
//                System.out.print(" * grad...[" + gradIndices[len - 3] + "][" + gradIndices[len - 2] + "][" + gradIndices[len - 1] + "]");
//                System.out.println(" -> " + inputVal + " * " + gradVal + " (=" + inputVal * gradVal + ")");
                g += inputVal * gradVal;
            }
        }
        return g;
    }

    public NDArray rot180(int yAxis, int xAxis) {
        return new NDArray(this.data, new JavaShapeRot180(this.shape, yAxis, xAxis));
    }

    public double[] getInternalData() {
        return data;
    }

    public NDArray softmax() {
        TMutableArray output = new TMutableArray(new double[this.data.length], shape);

        fillSoftMax(this, output, output.shape.newIndexArray(), 0);

        return output.migrateToImmutable();
    }

    public NDArray softMaxCrossEntropyGrad(NDArray softmax, NDArray oneHotArray) {
        return SoftmaxUtils.calcSoftmaxCrossEntropyGradient(softmax, oneHotArray).mul(this);
    }

    public DropOutResult dropOut(Random rnd, double dropoutKeep) {
        double gradValue = 1.0 / dropoutKeep;
        NDArray output = normalOrderedCopy();
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

        return new JavaDropOutResult(output, gradMaskData, dims);
    }

    public NDArray withUpdates(List<ValueUpdate> updates) {
        TMutableArray output = new TMutableArray(Arrays.copyOf(this.data, this.data.length), shape);
        for (ValueUpdate update : updates) {
            output.setAtOffset(update.offset, update.value);
        }
        return output.migrateToImmutable();
    }

    public NDArray clip(Double min, Double max) {
        NDArray copy = normalOrderedCopy();

        for (int i = 0; i < copy.data.length; i++) {
            if (min != null) {
                copy.data[i] = max(copy.data[i], min);
            }
            if (max != null) {
                copy.data[i] = min(copy.data[i], max);
            }
        }

        return copy;
    }

    public NDArray log() {
        NDArray copy = normalOrderedCopy();

        for (int i = 0; i < copy.data.length; i++) {
            copy.data[i] = Math.log(copy.data[i]);
        }

        return copy;
    }

    public JavaIntArray argmax(int axis) {
        validateAxisWithinBounds(shape, axis);

        NDArray src = normalOrderedCopy();

        Shape shape = src.shape;
        int safeAxis = shape.wrapNegIndex(axis);
        Shape outShape = shape.removeDimAt(safeAxis);

        int[] data = new int[toIntExact(outShape.size)];

        if (outShape.dimCount == 0) {
            data[0] = getMaxIndex(src, shape.newIndexArray(), safeAxis);
        } else {
            fillArgMax(src, shape.newIndexArray(), safeAxis, data, outShape, outShape.newIndexArray(), 0);
        }

        return new JavaIntArray(data, outShape);
    }

    private static void fillArgMax(NDArray src, int[] srcIndices, int axis, int[] tgt, Shape tgtShape, int[] tgtIndices, int tgtDim) {
        int len = tgtShape.at(tgtDim);
        for (int i = 0; i < len; i++) {
            tgtIndices[tgtDim] = i;
            srcIndices[tgtDim + (axis <= tgtDim ? 1 : 0)] = i;

            if (tgtDim == tgtIndices.length - 1) {
                int maxIndex = getMaxIndex(src, srcIndices, axis);
                int offset = tgtShape.calcDataIndex(tgtIndices);
                tgt[offset] = maxIndex;
            } else {
                fillArgMax(src, srcIndices, axis, tgt, tgtShape, tgtIndices, tgtDim + 1);
            }
        }
    }

    private static int getMaxIndex(NDArray src, int[] srcIndices, int axis) {
        int axisLen = src.shape.at(axis);
        double max = Double.NEGATIVE_INFINITY;
        int maxIndex = Integer.MIN_VALUE;
        for (int i = 0; i < axisLen; i++) {
            srcIndices[axis] = i;
            double srcV = src.dataAt(srcIndices);
            if (srcV > max) {
                max = srcV;
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public NDArray getAtIndicesOnAxis(JavaIntArray indices, int axis) {
        validateAxisWithinBounds(shape, axis);
        validateSameDimensionsExcept("indices", shape, indices.shape, axis);

        NDArray src = normalOrderedCopy();
        Shape shape = src.shape;

        int safeAxis = shape.wrapNegIndex(axis);

        Shape outShape = indices.shape;
        double[] data = new double[toIntExact(outShape.size)];

        if (outShape.dimCount > 0) {
            fillValueAtIndicesOnAxis(indices, src, shape.newIndexArray(), safeAxis,
                    data, outShape, outShape.newIndexArray(), 0);
        } else {
            data[0] = src.dataAt((Integer) indices.toInts());
        }

        return new NDArray(data, outShape);
    }

    private static void fillValueAtIndicesOnAxis(JavaIntArray valueIndices, NDArray src, int[] srcIndices, int axis,
                                                 double[] tgt, Shape tgtShape, int[] tgtIndices, int tgtDim) {
        int len = tgtShape.at(tgtDim);
        for (int i = 0; i < len; i++) {
            tgtIndices[tgtDim] = i;
            srcIndices[tgtDim + (axis <= tgtDim ? 1 : 0)] = i;

            if (tgtDim == tgtIndices.length - 1) {
                srcIndices[axis] = valueIndices.dataAt(tgtIndices);

                int offset = tgtShape.calcDataIndex(tgtIndices);
                tgt[offset] = src.dataAt(srcIndices);
            } else {
                fillValueAtIndicesOnAxis(valueIndices, src, srcIndices, axis, tgt, tgtShape, tgtIndices, tgtDim + 1);
            }
        }
    }

    public NDArray withUpdateAtIndicesOnAxis(JavaIntArray indices, int axis, NDArray change) {
        NDArray src = this.normalOrderedCopy();
        Shape shape = src.shape;

        validateAxisWithinBounds(shape, axis);
        validateSameDimensionsExcept("indices", shape, indices.shape, axis);
        validateSameDimensionsExcept("change", shape, change.shape, axis);


        int safeAxis = shape.wrapNegIndex(axis);
        int axisLen = shape.at(axis);

        double[] data = Arrays.copyOf(src.data, src.data.length);

        if (shape.dimCount == 1) {
            data[indices.data[0]] = change.data[0];
        } else {
            fillValuesIndicesOnAxis(indices, data, shape, shape.newIndexArray(), safeAxis, axisLen,
                    change, change.shape, change.shape.newIndexArray(), 0);
        }

        return new NDArray(data, shape);
    }

    private static void fillValuesIndicesOnAxis(JavaIntArray valueIndices, double[] tgt, Shape tgtShape, int[] tgtIndices,
                                                int axis, int axisLen,
                                                NDArray src, Shape srcShape, int[] srcIndices, int srcDim) {
        int len = srcShape.at(srcDim);
        for (int i = 0; i < len; i++) {
            srcIndices[srcDim] = i;
            tgtIndices[srcDim + (axis <= srcDim ? 1 : 0)] = i;

            if (srcDim == srcIndices.length - 1) {
                int axisIndex = valueIndices.dataAt(srcIndices);
                if (axisIndex >= 0 && axisIndex < axisLen) {
                    tgtIndices[axis] = axisIndex;
                    int offset = tgtShape.calcDataIndex(tgtIndices);
                    tgt[offset] = src.dataAt(srcIndices);
                } else {
                    throw new IndexOutOfBoundsException("Indices for axis " + axis + " must be in range [0," +
                            axisLen + "]: actual.indices" + Arrays.toString(tgtIndices) + "]=" + axisIndex);
                }
            } else {
                fillValuesIndicesOnAxis(valueIndices, tgt, tgtShape, tgtIndices, axis, axisLen, src, srcShape, srcIndices, srcDim + 1);
            }
        }
    }

    private static void fillSoftMax(NDArray src, TMutableArray tgt, int[] indices, int dim) {
        int len = tgt.shape.at(dim);
        if (indices.length - dim == 1) {
            //_mx = np.max(logits)
            double max = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < len; i++) {
                indices[dim] = i;
                double v = src.dataAt(indices);
                if (v > max) {
                    max = v;
                }
            }

            double expSum = 0;
            for (int i = 0; i < len; i++) {
                indices[dim] = i;
                //shifted = logits - _mx
                double shifted = src.dataAt(indices) - max;
                //l_exp = np.exp(shifted)
                double exped = exp(shifted);
                tgt.setAt(indices, exped);
                //l_exp_sum = np.sum(l_exp)
                expSum += exped;
            }

            for (int i = 0; i < len; i++) {
                indices[dim] = i;
                double exped = tgt.dataAt(indices);
                //l_sm = l_exp / l_exp_sum
                tgt.setAt(indices, exped / expSum);
            }
        } else {
            for (int i = 0; i < len; i++) {
                indices[dim] = i;
                fillSoftMax(src, tgt, indices, dim + 1);
            }
        }
    }

    public NDArray diag() {
        Shape outShape = shape.appendDim(shape.at(-1));

        double[] data = new double[toIntExact(outShape.size)];
        fillDiagonal(this, shape.newIndexArray(), data, outShape, outShape.newIndexArray(), 0);

        return new NDArray(data, outShape);
    }

    public NDArray concat(NDArray[] appendees, int axis) {
        int safeAxis = shape.wrapIndex(axis);

        NDArray[] srcs = toArray(this, appendees);
        Shape[] shapes = extractShapes(srcs);

        validateConcatShapes(shapes, safeAxis);
        Shape outShape = evalConcatShape(shapes, safeAxis);

        double[] data = new double[toIntExact(outShape.size)];

        fillConcat(srcs, shape.newIndexArray(),
                safeAxis, extractAxisLen(shapes, safeAxis),
                data, outShape, outShape.newIndexArray(), 0);

        return new NDArray(data, outShape);
    }

    private static NDArray[] toArray(NDArray firstElement, NDArray[] appendees) {
        NDArray[] copy = new NDArray[appendees.length + 1];
        System.arraycopy(appendees, 0, copy, 1, appendees.length);
        copy[0] = firstElement;
        return copy;
    }

    private static Shape[] getShapes(Shape thisShape, NDArray[] appendees) {
        Shape[] r = new Shape[appendees.length + 1];
        r[0] = thisShape;
        for (int i = 0; i < appendees.length; i++) {
            r[i + 1] = appendees[i].shape;
        }
        return r;
    }

    private static void fillConcat(NDArray[] srcs, int[] srcIndices,
                                   int axis, int[] axisLens,
                                   double[] data, Shape outShape, int[] outIndices, int dim) {
        int outLen = outShape.at(dim);

        for (int i = 0; i < outLen; i++) {
            outIndices[dim] = i;
            srcIndices[dim] = i;

            if (dim == outIndices.length - 1) {
                //... then is the last dimension
                int workingIndex = outIndices[axis];

                int srcIndex = 0;
                while (workingIndex >= axisLens[srcIndex]) {
                    workingIndex -= axisLens[srcIndex];
                    srcIndex++;
                }
                srcIndices[axis] = workingIndex;
                NDArray src = srcs[srcIndex];

                double val = src.dataAt(srcIndices);

                data[outShape.calcDataIndex(outIndices)] = val;
            } else {
                fillConcat(srcs, srcIndices, axis, axisLens, data, outShape, outIndices, dim + 1);
            }
        }
    }

    public List<NDArray> split(int axis, int[] axisLens) {
        Shape shape = this.shape;
        int safeAxis = shape.wrapNegIndex(axis);
        validateSplitLens(shape, safeAxis, axisLens);

        List<NDArray> parts = new ArrayList<>();
        int offset = 0;
        for (int axisLen : axisLens) {
            Shape outShape = evalSplitShape(shape, safeAxis, axisLen);
            double[] data = new double[toIntExact(outShape.size)];
            fillSplit(this, shape.newIndexArray(), safeAxis, offset, axisLen,
                    data, outShape, outShape.newIndexArray(), 0);

            parts.add(new NDArray(data, outShape));

            offset += axisLen;
        }
        return parts;
    }

    private static void fillSplit(NDArray src, int[] srcIndices,
                                  int axis, int offset, int axisLen,
                                  double[] data, Shape outShape, int[] outIndices,
                                  int dim) {
        int len, _offset;

        if (dim == axis) {
            len = axisLen;
            _offset = offset;
        } else {
            len = src.shape.at(dim);
            _offset = 0;
        }

        for (int i = 0; i < len; i++) {
            srcIndices[dim] = i + _offset;
            outIndices[dim] = i;

            if (dim == outIndices.length - 1) {
                //... then is the last dimension

                int outOffset = outShape.calcDataIndex(outIndices);
                data[outOffset] = src.dataAt(srcIndices);
            } else {
                fillSplit(src, srcIndices, axis, offset, axisLen, data, outShape, outIndices, dim + 1);
            }
        }
    }

    private static void fillDiagonal(NDArray src, int[] srcIndices, double[] out, Shape outShape, int[] outIndices, int srcDim) {
        int len = src.shape.at(srcDim);

        for (int i = 0; i < len; i++) {
            srcIndices[srcDim] = i;
            outIndices[srcDim] = i;

            if (srcDim == srcIndices.length - 1) {
                //... then is the last dimension
                // set last dim to same index, it's the diagonal
                outIndices[srcDim + 1] = i;
                int outOffset = outShape.calcDataIndex(outIndices);
                out[outOffset] = src.dataAt(srcIndices);
            } else {
                fillDiagonal(src, srcIndices, out, outShape, outIndices, srcDim + 1);
            }
        }
    }

    public NDArray subArray(int fromBatchIndex, int fromOffset, int endBatchIndex, int toOffset) {
        NDArray src = (shape.getClass() == Shape.class ? this : this.normalOrderedCopy());

        double[] data = new double[toOffset - fromOffset];
        System.arraycopy(src.data, fromOffset, data, 0, data.length);
        int[] dims = src.shape.toDimArray();
        dims[0] = endBatchIndex - fromBatchIndex;
        Shape outShape = new Shape(dims);

        return new NDArray(data, outShape);
    }

    public NDArray reshape(int... dims) {
        return new NDArray(this.data, this.shape.reshape(dims));
    }

    public NDArray reshape(Shape shape) {
        if (this.shape.getClass() != shape.getClass()) {
            throw new UnsupportedOperationException();
        }
        int[] dims = new int[shape.dimCount];
        for (int i = 0; i < dims.length; i++) {
            dims[i] = shape.at(i);
        }
        return reshape(dims);
    }

    public Object toDoubles() {
        return FlatToMultiDimArrayConverter.toDoubles(this.shape, i -> data[(int) i]);
    }

    public NDArray matmul(NDArray b) {
        return matmul(this, b);
    }

    public NDArray add(NDArray b) {
        return add(this, b);
    }

    private static NDArray add(NDArray a, NDArray b) {
        if (a.shape.dimCount == 0 &&
                b.shape.dimCount == 0) {
            return new NDArray(a.data[0] + b.data[0]);
        }
        if (a.shape.getClass() == b.shape.getClass() &&
                Arrays.equals(a.shape.dims, b.shape.dims)) {
            return fastAdd(a, b);
        }

        validateBroadcastShapes(a.shape, b.shape, -1);
        Shape outShape = evalBroadcastOutputShape(a.shape, b.shape);

        int[] indexArray = outShape.newIndexArray();
        double[] data = new double[outShape.size];

        add(a, b, data, outShape, indexArray, 0);

        return new NDArray(data, outShape);
    }

    private static NDArray fastAdd(NDArray a, NDArray b) {
        double[] data = copyOf(a.data, a.data.length);
        for (int i = data.length - 1; i >= 0; i--) {
            data[i] += b.data[i];
        }
        return new NDArray(data, a.shape/*.copy()*/);
    }

    public NDArray conv2d(NDArray filter, int offsetY, int offsetX) {
        return conv2d(this, filter, offsetY, offsetX);
    }

    private static NDArray conv2d(NDArray input, NDArray filter, int offsetY, int offsetX) {
        NDArray.validateConv2dShapes(input.shape, filter.shape);

        Shape outShape = evalConv2DShape(input.shape, filter.shape);
        double[] data = new double[outShape.size];

        double[] filledData = multiThreadingSupportRun(taskRange(0, input.shape.at(0)),
                range -> conv2dSegmentedAtFirstDim(range.start, range.end, input, filter, offsetY, offsetX,
                        data, outShape),
                (left, ignored) -> left);

        return new NDArray(filledData, outShape);
    }

    private static double[] conv2dSegmentedAtFirstDim(int start, int end,
                                                      NDArray input, NDArray filter,
                                                      int offsetY, int offsetX,
                                                      double[] data, Shape outShape) {
        int[] inIndices = input.shape.newIndexArray();
        int[] fIndices = filter.shape.newIndexArray();
        int[] outIndices = outShape.newIndexArray();
        for (int i = start; i < end; i++) {
            inIndices[0] = i;
            outIndices[0] = i;
            conv2dMain(input, inIndices, 1,
                    filter, fIndices,
                    offsetY, offsetX, data, outShape, outIndices);
        }
        return data;
    }

    private static void conv2dMain(NDArray input, int[] inIndices, int inDim,
                                   NDArray filter, int[] fIndices, int offsetY, int offsetX,
                                   double[] data, Shape outShape, int[] outIndices) {
        if (inIndices.length - inDim == 3) {
            int h = outShape.at(-3);
            int w = outShape.at(-2);
            // Eg.: in: <...,5,5,2> filter: <3,3,2,3>
            for (int y = 0; y < h; y++) {
                inIndices[inIndices.length - 3] = y;
                outIndices[outIndices.length - 3] = y;
                for (int x = 0; x < w; x++) {
                    inIndices[inIndices.length - 2] = x;
                    outIndices[outIndices.length - 2] = x;
                    conv2dAt(input, inIndices,
                            filter, fIndices, offsetY, offsetX,
                            data, outShape, outIndices);
                }
            }
        } else {
            int len = input.shape.at(inDim);
            for (int i = 0; i < len; i++) {
                inIndices[inDim] = i;
                outIndices[inDim] = i;
                conv2dMain(input, inIndices, inDim + 1,
                        filter, fIndices,
                        offsetY, offsetX, data, outShape, outIndices);
            }
        }
    }

    // Eg.: in: <...,5,5,2> filter: <3,3,2,3>
    private static void conv2dAt(NDArray input, int[] inIndices,
                                 NDArray filter, int[] fIndices, int offsetY, int offsetX,
                                 double[] data, Shape outShape, int[] outIndices) {
        int fLen = fIndices.length;
        int inLen = inIndices.length;
        int outLen = outIndices.length;
        int inYIdx = inLen - 3;
        int inXIdx = inLen - 2;
        int oldY = inIndices[inYIdx];
        int oldX = inIndices[inXIdx];
        int fH = filter.shape.at(0);
        int fW = filter.shape.at(1);
        int fYOffset = -(fH - 1) / 2 + offsetY;
        int fXOffset = -(fW - 1) / 2 + offsetX;
        int inCount = input.shape.at(-1);
        int outCount = outShape.at(-1);
        for (int outI = 0; outI < outCount; outI++) {
            fIndices[fLen - 1] = outI;
            double v = 0;
            for (int inI = 0; inI < inCount; inI++) {
                inIndices[inLen - 1] = inI;
                fIndices[fLen - 2] = inI;
                for (int y = 0; y < fH; y++) {
                    inIndices[inYIdx] = oldY + y + fYOffset;
                    fIndices[0] = y;
                    for (int x = 0; x < fW; x++) {
                        inIndices[inXIdx] = oldX + x + fXOffset;
                        fIndices[1] = x;
                        double iVal = input.dataAt(inIndices);
                        double fVal = filter.dataAt(fIndices);
                        v += iVal * fVal;
                    }
                }
            }
            outIndices[outLen - 1] = outI;
            int idx = outShape.calcDataIndex(outIndices);
            data[idx] = v;
        }

        inIndices[inYIdx] = oldY;
        inIndices[inXIdx] = oldX;
    }

    /**
     * @return 0 when out of bounds
     */
    public final double dataAt(int... indices) {
        if (indices.length == 1 && indices[0] == 0 && data.length == 1) {
            //scalar HACK
            return data[0];
        }
        if (shape.isValid(indices))
            return data[shape.calcDataIndex(indices)];
        else
            return 0; //OOB  //throw new UnsupportedOperationException();
    }

    public NDArray compare(JavaIntArray other, Comparison comparison, double trueValue, double falseValue) {
        Shape rightShape = other.shape;
        IntFunction<Double> left = offset -> this.data[offset];
        IntFunction<Double> right = offset -> (double) other.data[offset];

        return CompareHelper.compare(comparison::doubleIsTrue, trueValue, falseValue,
                this.shape, rightShape, left, right, new DoubleNDArrayWriter());
    }

    public NDArray compare(NDArray other, Comparison comparison, double trueValue, double falseValue) {
        IntFunction<Double> left = offset -> this.data[offset];
        IntFunction<Double> right = offset -> other.data[offset];

        return CompareHelper.compare(comparison::doubleIsTrue, trueValue, falseValue,
                this.shape, other.shape, left, right, new DoubleNDArrayWriter());
    }

    private static Shape evalConv2DShape(Shape input, Shape filter) {
        int[] dims = input.normalOrderedCopy().dims;
        dims[dims.length - 1] = filter.at(-1);
        return new Shape(dims);
    }

    private static NDArray matmul(NDArray a, NDArray b) {
        MatMulParams params = MatMulParams.expandSingleDimArrays(a.shape, b.shape, Shape::new);
        validateMatMulShapes(params.leftShape, params.rightShape);
        validateBroadcastShapes(params.leftShape, params.rightShape, -3);

        Shape outShape = evalMatMulShape(params.leftShape, params.rightShape);
        double[] data = new double[outShape.size];

        NDArray left = params.promoteLeft ? new NDArray(a.data, params.leftShape) : a;
        NDArray right = params.promoteRight ? new NDArray(b.data, params.rightShape) : b;

        int outputRows = outShape.at(-2);
        double[] filledData = multiThreadingSupportRun(
                taskRange(0, outputRows)
                        .withMinimumWorkLength(decideMinimumRowsPerThread(params.leftShape, outShape)),
                range -> matmul(range, left, right,
                        data, outShape, outShape.newIndexArray(), 0),
                (_left, ignored_) -> _left);

        Shape shape = params.revertDimExpandOfOutputShape(outShape);

        return new NDArray(filledData, shape);
    }

    public static int decideMinimumRowsPerThread(Shape leftShape, Shape outShape) {
        int valuesToMulPerOutput = leftShape.at(-1);
        int outputsPerRow = outShape.at(-1);

        return 1 + 512 / (valuesToMulPerOutput * outputsPerRow);
    }

    public NDArray normalOrderedCopy() {
        Shape tgtShape = this.shape.normalOrderedCopy();
        double[] data = this.shape.convertDataToShape(this.data, tgtShape);

        return new NDArray(data, tgtShape);
    }

    private NDArray expandDims(int... indicesForSingleDims) {
        if (shape instanceof ReorderedJavaShape) {
            return expandDims(this.normalOrderedCopy(), indicesForSingleDims);
        }
        return expandDims(this, indicesForSingleDims);
    }

    private static NDArray expandDims(NDArray m, int... indicesForSingleDims) {
        int[] _tmp = copyOf(indicesForSingleDims, indicesForSingleDims.length);
        for (int i = 0; i < _tmp.length; i++) {
            if (_tmp[i] <= -1)
                _tmp[i] += m.shape.dimCount + 1;
        }
        sort(_tmp);
        List<Integer> dims = new ArrayList<>();
        for (int i = 0; i < m.shape.dimCount; i++) {
            dims.add(m.shape.at(i));
        }
        for (int i = _tmp.length - 1; i >= 0; i--) {
            dims.add(_tmp[i], 1);
        }
        int[] dimArr = new int[dims.size()];
        for (int i = 0; i < dims.size(); i++) {
            dimArr[i] = dims.get(i);
        }

        return new NDArray(m.data, new Shape(dimArr));
    }

    private static void add(NDArray a, NDArray b,
                            double[] out, Shape outShape,
                            int[] indices, int dim) {
        if (indices.length - dim == 1) {
            //...the last index
            int w = outShape.at(-1);
            int dims = indices.length;
            for (int x = 0; x < w; x++) {
                indices[dims - 1] = x;
                double _a = a.getBroadcasted(indices);
                double _b = b.getBroadcasted(indices);
                int outIndex = outShape.calcDataIndex(indices);
                out[outIndex] = _a + _b;
            }
        } else {
            int at = outShape.at(dim);
            for (int i = 0; i < at; i++) {
                indices[dim] = i;
                add(a, b, out, outShape, indices, dim + 1);
            }
        }
    }

    public static double[] matmul(MultiThreadingSupport.TaskRange rowRange,
                                  NDArray a, NDArray b,
                                  double[] out, Shape outShape,
                                  int[] indices, int dim) {

        if (indices.length - dim == 2) {
            //...is second last index
            int w = outShape.at(-1);
            int valCount = a.shape.at(-1);
            int dims = indices.length;
            for (int y = rowRange.start; y < rowRange.end; y++) {
                for (int x = 0; x < w; x++) {
                    double v = 0;
                    for (int i = 0; i < valCount; i++) {
                        indices[dims - 2] = y;
                        indices[dims - 1] = i;
                        double _a = a.getBroadcasted(indices);
                        indices[dims - 2] = i;
                        indices[dims - 1] = x;
                        double _b = b.getBroadcasted(indices);
                        v += _a * _b;
                    }
                    indices[dims - 2] = y;
                    indices[dims - 1] = x;
                    int outIndex = outShape.calcDataIndex(indices);
                    out[outIndex] = v;
                }
            }
        } else {
            int at = outShape.at(dim);
            for (int i = 0; i < at; i++) {
                indices[dim] = i;
                matmul(rowRange, a, b, out, outShape, indices, dim + 1);
            }
        }
        return out;
    }

    private double getBroadcasted(int[] indices) {
        return data[shape.getBroadcastOffset(indices)];
    }

    public static Shape evalBroadcastOutputShape(Shape a, Shape b) {
        return new Shape(createBroadcastResultDims(a, b));
    }

    public static Shape evalMatMulShape(Shape a, Shape b) {
        int[] dims = evalMatMulResultDims(a, b);
        return new Shape(dims);
    }

    public NDArray transpose(int... axes) {
        ReorderedJavaShape shape = axes.length == 0 ?
                ReorderedJavaShape.reverseOf(this.shape) : ReorderedJavaShape.customOrder(this.shape, axes);
        return new NDArray(data, shape);
    }

    public NDArray sum() {
        boolean[] toCollapse = new boolean[shape.dimCount];
        fill(toCollapse, true);
        return sum(toCollapse, REMOVE_DIM);
    }

    public JavaMaxPool2dResult maxPool2d(int size) {
        Shape outShape = getMaxPool2dResultShape(shape, size);

        TMutableArray tgt = new TMutableArray(new double[outShape.size], outShape);

        int[] inputIndices = shape.newIndexArray();
        int[] tgtIndices = outShape.newIndexArray();
        Shape maxIndexShape = createMax2dIndexShape(outShape);
        int[] tmpMaxIndices = maxIndexShape.newIndexArray();
        int[] maxIndexData = new int[maxIndexShape.size];

        fillMax2d(inputIndices, size, tgt, tgtIndices,
                maxIndexShape, maxIndexData, tmpMaxIndices,
                0);

        return new JavaMaxPool2dResult(tgt.migrateToImmutable(), shape, maxIndexShape, maxIndexData);
    }

    private void fillMax2d(int[] inputIndices, int size,
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

                        double maxVal = getMax2dVal(inputIndices,
                                y * size, x * size, c, size,
                                maxIndexShape, maxIndexData, tmpMaxIndices);

                        tgt.setAt(tgtIndices, maxVal);
                    }
                }
            }
        } else {
            int len = shape.at(dim);
            for (int i = 0; i < len; i++) {
                inputIndices[dim] = i;
                tgtIndices[dim] = i;
                tmpMaxIndices[dim] = i;

                fillMax2d(inputIndices, size, tgt, tgtIndices,
                        maxIndexShape, maxIndexData, tmpMaxIndices,
                        dim + 1);
            }
        }
    }

    private double getMax2dVal(int[] inputIndices,
                               int yInputOffset, int xInputOffset, int c,
                               int size,
                               Shape maxIndexShape, int[] maxIndexData, int[] tmpMaxIndices) {
        double max = Double.NEGATIVE_INFINITY;
        int len = inputIndices.length;
        inputIndices[len - 1] = c;
        int maxY = -1;
        int maxX = -1;

        int inputH = shape.at(-3);
        int inputW = shape.at(-2);

        for (int y = 0; y < size; y++) {
            int inY = y + yInputOffset;

            if (inY < inputH) {
                inputIndices[len - 3] = inY;
                for (int x = 0; x < size; x++) {
                    int inX = x + xInputOffset;
                    if (inX < inputW) {
                        inputIndices[len - 2] = inX;
                        double inVal = dataAt(inputIndices);
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

    public NDArray maxPool2dGrad(MaxPool2dResult result) {
        JavaMaxPool2dResult r = (JavaMaxPool2dResult) result;
        return distribute2dMaxGrad(this, r.inputShape, r.maxIndexShape, r.maxIndexData);
    }

    public ReluResult relu(double leakyScale) {
        NDArray copy = normalOrderedCopy();
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

        return new JavaReluResult(data, copy.shape, gradMaskData);
    }

    private static Shape createMax2dIndexShape(Shape outShape) {
        int[] idxDims = copyOf(outShape.toDimArray(), outShape.dimCount + 1);
        // for (y,x) pair to log the original location of the value in
        // the input matrix.
        idxDims[outShape.dimCount] = 2;
        return new Shape(idxDims);
    }

    public NDArray sum(boolean[] dimsToCollapse, DimKeepRemove keepRemove) {
        if (dimsToCollapse.length != shape.dimCount) {
            throw new RuntimeException("input collapse dims must have same length as shape");
        }
        Shape physicalShape = toPhysicalShape(shape, dimsToCollapse);
        int[] dimMapping = createSrcToTargetMapping(dimsToCollapse);

        double[] target = new double[toIntExact(physicalShape.size)];
        sum(data, shape, shape.newIndexArray(), 0,
                target, physicalShape, physicalShape.newIndexArray(), dimMapping);

        if (keepRemove == DimKeepRemove.KEEP_DIM) {
            return new NDArray(target, toPhysicalShapeWithKeep(shape, dimsToCollapse));
        }
        return new NDArray(target, physicalShape);
    }

    public void sum(NDArray x, boolean[] dimsToCollapse, DimKeepRemove keepRemove) {
//        if (dimsToCollapse.length != shape.dimCount)
//            throw new RuntimeException("input collapse dims must have same length as shape");

        //Shape physicalShape = toPhysicalShape(shape, dimsToCollapse);
        int[] dimMapping = createSrcToTargetMapping(dimsToCollapse);

        //double[] t = new double[toIntExact(physicalShape.getSize())];
        sum(x.data, x.shape, 0,
                data, shape, dimMapping);
    }

    private static void sum(double[] src, Shape srcShape, int srcI,
                            double[] tgt, Shape tgtShape,
                            int[] srcToTgtMapping) {
        sum(src, srcShape, srcShape.newIndexArray(), srcI,
                tgt, tgtShape, tgtShape.newIndexArray(), srcToTgtMapping);
    }

    private static void sum(double[] src, Shape srcShape, int[] srcIndices, int srcI,
                            double[] tgt, Shape tgtShape, int[] tgtIndices,
                            int[] srcToTgtMapping) {
        int dimLen = srcShape.at(srcI);
        if (srcI == srcIndices.length - 1) {
            //...last dim
            for (int i = 0; i < dimLen; i++) {
                srcIndices[srcI] = i;
                double srcVal = src[srcShape.calcDataIndex(srcIndices)];
                int srcToTgt = srcToTgtMapping[srcI];
                if (srcToTgt != -1) {
                    tgtIndices[srcToTgt] = i;
                }
                int tgtOffset = tgtShape.calcDataIndex(tgtIndices);
                tgt[tgtOffset] += srcVal;
            }
        } else {
            for (int i = 0; i < dimLen; i++) {
                srcIndices[srcI] = i;
                int srcToTgt = srcToTgtMapping[srcI];
                if (srcToTgt != -1) {
                    tgtIndices[srcToTgt] = i;
                }
                sum(src, srcShape, srcIndices, srcI + 1,
                        tgt, tgtShape, tgtIndices,
                        srcToTgtMapping);
            }
        }
    }

    private static int[] createSrcToTargetMapping(boolean[] dimsToCollapse) {
        int[] mapping = new int[dimsToCollapse.length];
        fill(mapping, -1);
        int idx = 0;
        for (int i = 0; i < dimsToCollapse.length; i++) {
            boolean collapseDimension = dimsToCollapse[i];
            if (!collapseDimension) {
                mapping[i] = idx;
                idx++;
            }
        }
        return mapping;
    }

    public static Shape toPhysicalShape(Shape shape, boolean[] dimsToCollapse) {
        int count = countFalse(dimsToCollapse);
        int[] physicalDims = new int[count];
        int idx = 0;
        for (int i = 0; i < dimsToCollapse.length; i++) {
            if (!dimsToCollapse[i]) {
                physicalDims[idx] = shape.at(i);
                idx++;
            }
        }
        return ProviderStore.shape(physicalDims);
    }

    private static Shape toPhysicalShapeWithKeep(Shape shape, boolean[] dimsToCollapse) {
        int[] physicalDims = new int[dimsToCollapse.length];
        fill(physicalDims, 1);
        for (int i = 0; i < dimsToCollapse.length; i++) {
            if (!(dimsToCollapse[i])) {
                physicalDims[i] = shape.at(i);
            }
        }
        return new Shape(physicalDims);
    }

    private static int countFalse(boolean[] dimsToCollapse) {
        int count = 0;
        for (Boolean c : dimsToCollapse) if (!(c != null && c)) count++;
        return count;
    }

    public NDArray negate() {
        if (this.shape instanceof ReorderedJavaShape)
            throw new UnsupportedOperationException("reordered shape not yet supported");

        double[] data = this.data.clone();
        for (int i = 0; i < data.length; i++)
            data[i] = -data[i];

        return new NDArray(data, shape);
    }

    public NDArray sqr() {
        double[] cp = copyOf(data, data.length);
        for (int i = 0; i < cp.length; i++) {
            cp[i] *= cp[i];
        }

        return new NDArray(cp, shape.copy());
    }

    public NDArray pow(double power) {
        double[] data = copyOf(this.data, this.data.length);
        double[] filledData = multiThreadingSupportRun(taskRange(0, this.data.length, 64),
                range -> pow(range.start, range.end, data, power),
                (left, ignored) -> left);

        return new NDArray(filledData, shape.copy());
    }

    private static double[] pow(int start, int end, double[] data, double power) {
        for (int i = start; i < end; i++) {
            data[i] = Math.pow(data[i], power);
        }
        return data;
    }

    public NDArray sqrt() {
        double[] data = copyOf(this.data, this.data.length);
        double[] filledData = multiThreadingSupportRun(taskRange(0, data.length, 64),
                range -> sqrt(range.start, range.end, data),
                (left, ignored) -> left);

        return new NDArray(filledData, shape.copy());
    }

    private static double[] sqrt(int start, int end, double[] data) {
        for (int i = start; i < end; i++) {
            data[i] = Math.sqrt(data[i]);
        }
        return data;
    }

    public NDArray div(NDArray b) {
        return div(this, b);
    }

    private static NDArray div(NDArray a, NDArray b) {
        if (a.shape.getClass() == b.shape.getClass() &&
                Arrays.equals(a.shape.dims, b.shape.dims)) {
            return fastDiv(a, b);
        }

        validateBroadcastShapes(a.shape, b.shape, -1);
        Shape outShape = evalBroadcastOutputShape(a.shape, b.shape);

        int[] indexArray = outShape.newIndexArray();
        double[] data = new double[outShape.size];

        div(a, b,
                data, outShape,
                indexArray, 0);

        return new NDArray(data, outShape);
    }

    private static NDArray fastDiv(NDArray a, NDArray b) {
        double[] data = copyOf(a.data, a.data.length);
        for (int i = data.length - 1; i >= 0; i--) {
            data[i] /= b.data[i];
        }
        return new NDArray(data, a.shape.copy());
    }

    private static void div(NDArray a, NDArray b,
                            double[] out, Shape outShape,
                            int[] indices, int dim) {
        if (indices.length - dim == 1) {
            //...the last index
            int w = outShape.at(-1);
            int dims = indices.length;
            for (int x = 0; x < w; x++) {
                indices[dims - 1] = x;
                double _a = a.getBroadcasted(indices);
                double _b = b.getBroadcasted(indices);
                int outIndex = outShape.calcDataIndex(indices);
                out[outIndex] = _a / _b;
            }
        } else {
            int at = outShape.at(dim);
            for (int i = 0; i < at; i++) {
                indices[dim] = i;
                div(a, b, out, outShape, indices, dim + 1);
            }
        }
    }

    public NDArray mul(NDArray b) {
        return mul(this, b);
    }

    private static NDArray mul(NDArray a, NDArray b) {
        if (a.shape.getClass() == b.shape.getClass() &&
                Arrays.equals(a.shape.dims, b.shape.dims)) {
            return fastMul(a, b);
        }

        validateBroadcastShapes(a.shape, b.shape, -1);
        Shape outShape = evalBroadcastOutputShape(a.shape, b.shape);

        int[] indexArray = outShape.newIndexArray();
        double[] data = new double[outShape.size];

        mul(a, b,
                data, outShape,
                indexArray, 0);

        return new NDArray(data, outShape);
    }

    private static NDArray fastMul(NDArray a, NDArray b) {
        double[] data = copyOf(a.data, a.data.length);
        for (int i = data.length - 1; i >= 0; i--) {
            data[i] *= b.data[i];
        }
        return new NDArray(data, a.shape.copy());
    }

    private static void mul(NDArray a, NDArray b,
                            double[] out, Shape outShape,
                            int[] indices, int dim) {
        if (indices.length - dim == 1) {
            //...the last index
            int w = outShape.at(-1);
            int dims = indices.length;
            for (int x = 0; x < w; x++) {
                indices[dims - 1] = x;
                double _a = a.getBroadcasted(indices);
                double _b = b.getBroadcasted(indices);
                int outIndex = outShape.calcDataIndex(indices);
                out[outIndex] = _a * _b;
            }
        } else {
            int at = outShape.at(dim);
            for (int i = 0; i < at; i++) {
                indices[dim] = i;
                mul(a, b, out, outShape, indices, dim + 1);
            }
        }
    }

    public NDArray div(double v) {
        double[] data = copyOf(this.data, this.data.length);
        for (int i = 0; i < data.length; i++) {
            data[i] /= v;
        }
        return new NDArray(data, shape.copy());
    }

    public static NDArray conv2d(NDArray filter, int offsetY, int offsetX, int outHeight, int outWidth) {
        throw new UnsupportedOperationException();
    }

    public NDArray mul(double v) {
        double[] data = copyOf(this.data, this.data.length);
        double[] filledData = multiThreadingSupportRun(taskRange(0, data.length, 64),
                range -> mul(range.start, range.end, data, v),
                (left, ignored) -> left);

        return new NDArray(filledData, shape.copy());
    }

    private static double[] mul(int start, int end, double[] data, double v) {
        for (int i = start; i < end; i++) {
            data[i] *= v;
        }
        return data;
    }

    public NDArray add(double v) {
        double[] data = copyOf(this.data, this.data.length);
        for (int i = 0; i < data.length; i++) {
            data[i] += v;
        }
        return new NDArray(data, shape.copy());
    }

    @Override
    public String toString() {
        return StringUtils.toString(this);
    }

    public NDArray sub(NDArray m) {
        return add(m.negate());
    }

    public NDArray conv2d(NDArray filter) {
        return conv2d(filter, 0, 0);
    }

    @Override
    public void prepareDependenciesForDisposal() {
        waitForValueReady();
    }

    public void waitForValueReady() {
        // do nothing
    }

    public NDArray sum(DimKeepRemove keepRemove, int... axes) {
        Shape inputShape = shape;
        boolean[] dimsToCollapse = inputShape.newCollapseArray();
        for (int axis : axes)
            dimsToCollapse[inputShape.wrapNegIndex(axis)] = true;

        return sum(dimsToCollapse, keepRemove);
    }

    public NDArray sumFirstDims(int firstDimsToRemove, DimKeepRemove keepRemove) {
        boolean[] dimsToCollapse = new boolean[shape.dimCount];
        Arrays.fill(dimsToCollapse, false);
        for (int i = 0; i < firstDimsToRemove; i++)
            dimsToCollapse[i] = true;

        return sum(dimsToCollapse, keepRemove);
    }

    @Override
    public void dispose() {
        // do nothing
    }

    public NDArray transposeLast2D() {
        int dimCount = shape.dimCount;
        if (dimCount <= 1) {
            throw new DimensionMissing("Expected 2+ dimensions: actualDim=" + dimCount);
        }
        int[] axes = new int[dimCount];
        for (int i = 0; i < axes.length; i++) {
            axes[i] = i;
        }
        axes[dimCount - 2] = dimCount - 1;
        axes[dimCount - 1] = dimCount - 2;

        return transpose(axes);
    }

    public NDArray concat(NDArray appendee, int axis) {
        return concat(new NDArray[]{appendee}, axis);
    }


    public NDArray subBatch(int batchId, int batchSize) {
        Shape shape = this.shape;

        int[] indices = shape.newIndexArray();
        int fromBatchIndex = batchId * batchSize;
        indices[0] = fromBatchIndex;
        int fromOffset = shape.calcDataIndex(indices);
        int endBatchIndex = min(shape.at(0), (batchId + 1) * batchSize);
        indices[0] = endBatchIndex;
        int toOffset = shape.calcDataIndex(indices);

        return subArray(fromBatchIndex, fromOffset, endBatchIndex, toOffset);
    }

    public void zero() {
        Arrays.fill(data, 0);
    }

    public double scalar() {
        if (data.length != 1)
            throw new UnsupportedOperationException(this + " is not a scalar");
        return dataAt(0);
    }

    public enum DimKeepRemove {
        REMOVE_DIM {
            @Override
            public Shape toActualOutShape(Shape inShape, Shape outShapeWithSingleDimensions, Boolean[] dimsToSum) {
                int dimCount = countFalse(dimsToSum);
                int[] dims = new int[dimCount];
                int t = 0;

                int orgCount = inShape.dimCount;
                for (int i = 0; i < orgCount; i++) {
                    if (!dimsToSum[i]) {
                        dims[t++] = inShape.at(i);
                    }
                }

                return ProviderStore.shape(dims);
            }

            private static int countFalse(Boolean[] dimsToSum) {
                int c = 0;
                for (Boolean d : dimsToSum)
                    if (!d)
                        c++;
                return c;
            }
        },
        KEEP_DIM {
            @Override
            public Shape toActualOutShape(Shape inShape, Shape outShapeWithSingleDimensions, Boolean[] dimsToSum) {
                return outShapeWithSingleDimensions;
            }
        };

        public abstract Shape toActualOutShape(Shape inShape, Shape outShapeWithSingleDimensions, Boolean[] dimsToSum);
    }

    public interface InternalIntReader {
        int readValue(long index);
    }

    public interface InternalDoubleReader {
        double readValue(long index);
    }

    public interface MaxPool2dResult {
        NDArray output();
    }

    public interface ReluResult {
        NDArray getOutput();

        NDArray createMask();
    }

    public interface DropOutResult {
        NDArray output();

        NDArray createMask();
    }

    private record JavaMaxPool2dResult(NDArray output, Shape inputShape, Shape maxIndexShape,
                                       int[] maxIndexData) implements MaxPool2dResult {
    }

    private record JavaReluResult(double[] data, Shape shape, double[] gradMaskData) implements ReluResult {

        @Override
        public NDArray getOutput() {
            return new NDArray(data, shape);
        }

        @Override
        public NDArray createMask() {
            return new NDArray(gradMaskData, shape);
        }
    }

    private record JavaDropOutResult(NDArray output, double[] gradMaskData, int[] dims) implements DropOutResult {

        @Override
        public NDArray createMask() {
            return new NDArray(gradMaskData, Shape.of(dims));
        }
    }

    private static class DoubleNDArrayWriter implements CompareHelper.CompareWriter<Double, NDArray> {
        private double[] data;

        @Override
        public NDArray scalar(Double value) {
            return new NDArray(value);
        }

        @Override
        public NDArray toArray(Shape shape) {
            return new NDArray(data, shape);
        }

        @Override
        public void prepareDate(int size) {
            data = new double[size];
        }

        @Override
        public void write(int offset, Double outVal) {
            data[offset] = outVal;
        }
    }

}
