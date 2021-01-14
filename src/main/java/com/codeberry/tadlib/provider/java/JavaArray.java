package com.codeberry.tadlib.provider.java;

import com.codeberry.tadlib.array.NDArray;
import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.array.util.FlatToMultiDimArrayConverter;
import com.codeberry.tadlib.array.util.SoftmaxUtils;
import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.util.MultiThreadingSupport;
import com.codeberry.tadlib.util.StringUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static com.codeberry.tadlib.array.NDArray.DimKeepRemove.REMOVE_DIM;
import static com.codeberry.tadlib.array.util.DimensionUtils.*;
import static com.codeberry.tadlib.util.MultiThreadingSupport.TaskRange.taskRange;
import static com.codeberry.tadlib.util.MultiThreadingSupport.multiThreadingSupportRun;
import static java.lang.Boolean.TRUE;
import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.util.Arrays.*;

public class JavaArray implements NDArray {
    public static final JavaArray ZERO = new JavaArray(0.0);

    private final double[] data;
    public final JavaShape shape;

    public JavaArray(double val) {
        this(new double[]{val}, JavaShape.zeroDim());
    }

    public JavaArray(double[] data) {
        this(data, new JavaShape(data.length));
    }

    public JavaArray(double[] data, JavaShape shape) {
        this.data = data;
        this.shape = shape;
    }

    public static JavaArray distribute2dMaxGrad(JavaArray grad, JavaShape inputShape, JavaShape maxIndexShape, int[] maxIndexData) {
        TMutableArray outputGrad = new TMutableArray(inputShape);

        int[] tmpOutputGradIndices = outputGrad.shape.newIndexArray();
        int[] tmpGradIndices = grad.shape.newIndexArray();
        int[] tmpMaxIndices = maxIndexShape.newIndexArray();
        fillMax2dGradInto(outputGrad, maxIndexShape, maxIndexData, grad, 0,
                tmpOutputGradIndices, tmpGradIndices, tmpMaxIndices);

        return outputGrad.migrateToImmutable();
    }

    private static void fillMax2dGradInto(TMutableArray outputGrad, JavaShape maxIndexShape, int[] maxIndexData, JavaArray grad, int dim,
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
        int len = predicted.getShape().at(dim);
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

    public JavaArray calcConv2dFilterGradient(NDArray input, NDArray filter) {
        JavaShape filterShape = (JavaShape) filter.getShape();
        int[] dims = new int[filterShape.dimCount + 1];
        System.arraycopy(filterShape.toDimArray(), 0, dims, 1, filterShape.dimCount);
        dims[0] = input.getShape().at(0);
        JavaShape tgtShape = new JavaShape(dims);

        JavaArray gradPerInputExample = multiThreadingSupportRun(taskRange(0, shape.at(0)),
                range -> accumulateFilterGradientAtFirstDim(range, (JavaArray) input, tgtShape),
                (left, right) -> left.add(right));

        return (JavaArray) gradPerInputExample.sumFirstDims(1, REMOVE_DIM);
    }

    private JavaArray accumulateFilterGradientAtFirstDim(MultiThreadingSupport.TaskRange range, JavaArray input, JavaShape tgtShape) {
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
                                          JavaArray input, int[] inIndices,
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

    @Override
    public NDArray rot180(int yAxis, int xAxis) {
        return new JavaArray(this.data, new JavaShapeRot180(this.shape, yAxis, xAxis));
    }

    public double[] getInternalData() {
        return data;
    }

    public JavaArray softmax() {
        TMutableArray output = new TMutableArray(new double[this.data.length], shape);

        fillSoftMax(this, output, output.shape.newIndexArray(), 0);

        return output.migrateToImmutable();
    }

    @Override
    public NDArray softMaxGrad(NDArray softmax, NDArray oneHotArray) {
        List<ValueUpdate> updates = SoftmaxUtils.getSoftmaxGradientUpdates(softmax, softmax.getShape().newIndexArray(),
                oneHotArray, 0);

        return softmax.withUpdates(updates).mul(this);
    }

    @Override
    public DropOutResult dropOut(Random rnd, double dropoutKeep) {
        double gradValue = 1.0 / dropoutKeep;
        JavaArray output = normalOrderedCopy();
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

    @Override
    public NDArray withUpdates(List<ValueUpdate> updates) {
        TMutableArray output = new TMutableArray(Arrays.copyOf(this.data, this.data.length), shape);
        for (ValueUpdate update : updates) {
            output.setAtOffset(update.offset, update.value);
        }
        return output.migrateToImmutable();
    }

    @Override
    public NDArray clip(Double min, Double max) {
        JavaArray copy = normalOrderedCopy();

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

    @Override
    public NDArray log() {
        JavaArray copy = normalOrderedCopy();

        for (int i = 0; i < copy.data.length; i++) {
            copy.data[i] = Math.log(copy.data[i]);
        }

        return copy;
    }

    private static void fillSoftMax(JavaArray src, TMutableArray tgt, int[] indices, int dim) {
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
                double exped = Math.exp(shifted);
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

    public NDArray subArray(int fromBatchIndex, int fromOffset, int endBatchIndex, int toOffset) {
        JavaArray src = (shape.getClass() == JavaShape.class ? this : this.normalOrderedCopy());

        double[] data = new double[toOffset - fromOffset];
        System.arraycopy(src.data, fromOffset, data, 0, data.length);
        int[] dims = src.shape.toDimArray();
        dims[0] = endBatchIndex - fromBatchIndex;
        JavaShape outShape = new JavaShape(dims);

        return new JavaArray(data, outShape);
    }

    public JavaArray reshape(int... dims) {
        return new JavaArray(this.data, this.shape.reshape(dims));
    }

    public JavaArray reshape(JavaShape shape) {
        return (JavaArray) reshape((Shape) shape);
    }

    @Override
    public Object toDoubles() {
        return FlatToMultiDimArrayConverter.toDoubles(this.shape, i -> data[(int) i]);
    }

    @Override
    public NDArray matmul(NDArray b) {
        return matmul((JavaArray) b);
    }

    public JavaArray matmul(JavaArray b) {
        return matmul(this, b);
    }

    public JavaArray add(NDArray b) {
        return add(this, (JavaArray) b);
    }

    private static JavaArray add(JavaArray a, JavaArray b) {
        if (a.shape.dimCount == 0 &&
                b.shape.dimCount == 0) {
            return new JavaArray(a.data[0] + b.data[0]);
        }
        if (a.shape.getClass() == b.shape.getClass() &&
                Arrays.equals(a.shape.dims, b.shape.dims)) {
            return fastAdd(a, b);
        }

        validateBroadcastShapes(a.shape, b.shape, -1);
        JavaShape outShape = evalBroadcastOutputShape(a.shape, b.shape);

        int[] indexArray = outShape.newIndexArray();
        double[] data = new double[outShape.size];

        add(a, b, data, outShape, indexArray, 0);

        return new JavaArray(data, outShape);
    }

    private static JavaArray fastAdd(JavaArray a, JavaArray b) {
        double[] data = copyOf(a.data, a.data.length);
        for (int i = data.length - 1; i >= 0; i--) {
            data[i] += b.data[i];
        }
        return new JavaArray(data, a.shape.copy());
    }

    @Override
    public JavaShape getShape() {
        return shape;
    }

    @Override
    public JavaArray conv2d(NDArray filter, int offsetY, int offsetX) {
        return conv2d(this, (JavaArray) filter, offsetY, offsetX);
    }

    private static JavaArray conv2d(JavaArray input, JavaArray filter, int offsetY, int offsetX) {
        NDArray.validateConv2dShapes(input.shape, filter.shape);

        JavaShape outShape = evalConv2DShape(input.shape, filter.shape);
        double[] data = new double[outShape.size];

        double[] filledData = multiThreadingSupportRun(taskRange(0, input.shape.at(0)),
                range -> conv2dSegmentedAtFirstDim(range.start, range.end, input, filter, offsetY, offsetX,
                        data, outShape),
                (left, ignored) -> left);

        return new JavaArray(filledData, outShape);
    }

    private static double[] conv2dSegmentedAtFirstDim(int start, int end,
                                                      JavaArray input, JavaArray filter,
                                                      int offsetY, int offsetX,
                                                      double[] data, JavaShape outShape) {
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

    private static void conv2dMain(JavaArray input, int[] inIndices, int inDim,
                                   JavaArray filter, int[] fIndices, int offsetY, int offsetX,
                                   double[] data, JavaShape outShape, int[] outIndices) {
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
    private static void conv2dAt(JavaArray input, int[] inIndices,
                                 JavaArray filter, int[] fIndices, int offsetY, int offsetX,
                                 double[] data, JavaShape outShape, int[] outIndices) {
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
    public double dataAt(int... indices) {
        if (shape.isValid(indices)) {
            int idx = shape.calcDataIndex(indices);
            return data[idx];
        }
        return 0;
    }

    private static JavaShape evalConv2DShape(JavaShape input, JavaShape filter) {
        int[] dims = input.normalOrderedCopy().dims;
        dims[dims.length - 1] = filter.at(-1);
        return new JavaShape(dims);
    }

    private static JavaArray matmul(JavaArray a, JavaArray b) {
        MatMulParams params = MatMulParams.expandSingleDimArrays(a.shape, b.shape, JavaShape::new);

        validateMatMulShapes(params.leftShape, params.rightShape);
        validateBroadcastShapes(params.leftShape, params.rightShape, -3);

        JavaShape outShape = evalMatMulShape(params.leftShape, params.rightShape);
        double[] data = new double[outShape.size];

        JavaArray left = params.promoteLeft ? new JavaArray(a.data, (JavaShape) params.leftShape) : a;
        JavaArray right = params.promoteRight ? new JavaArray(b.data, (JavaShape) params.rightShape) : b;

        int outputRows = outShape.at(-2);
        double[] filledData = multiThreadingSupportRun(
                taskRange(0, outputRows)
                        .withMinimumWorkLength(decideMinimumRowsPerThread(params.leftShape, outShape)),
                range -> matmul(range, left, right,
                        data, outShape, outShape.newIndexArray(), 0),
                (_left, ignored_) -> _left);

        return new JavaArray(filledData,
                (JavaShape) params.revertDimExpandOfOutputShape(outShape));
    }

    private static int decideMinimumRowsPerThread(Shape leftShape, Shape outShape) {
        int valuesToMulPerOutput = leftShape.at(-1);
        int outputsPerRow = outShape.at(-1);

        return 1 + 512 / (valuesToMulPerOutput * outputsPerRow);
    }

    public JavaArray normalOrderedCopy() {
        JavaShape tgtShape = this.shape.normalOrderedCopy();
        double[] data = this.shape.convertDataToShape(this.data, tgtShape);

        return new JavaArray(data, tgtShape);
    }

    private JavaArray expandDims(int... indicesForSingleDims) {
        if (shape instanceof ReorderedJavaShape) {
            return expandDims(this.normalOrderedCopy(), indicesForSingleDims);
        }
        return expandDims(this, indicesForSingleDims);
    }

    private static JavaArray expandDims(JavaArray m, int... indicesForSingleDims) {
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

        return new JavaArray(m.data, new JavaShape(dimArr));
    }

    private static void add(JavaArray a, JavaArray b,
                            double[] out, JavaShape outShape,
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

    private static double[] matmul(MultiThreadingSupport.TaskRange rowRange,
                                   JavaArray a, JavaArray b,
                                   double[] out, JavaShape outShape,
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
        int offset = shape.getBroadcastOffset(indices);

        return data[offset];
    }

    private static JavaShape evalBroadcastOutputShape(JavaShape a, JavaShape b) {
        return new JavaShape(createBroadcastResultDims(a, b));
    }

    private static JavaShape evalMatMulShape(Shape a, Shape b) {
        int[] dims = evalMatMulResultDims(a, b);
        return new JavaShape(dims);
    }

    @Override
    public JavaArray transpose(int... axes) {
        ReorderedJavaShape shape = axes.length == 0 ?
                ReorderedJavaShape.reverseOf(this.shape) : ReorderedJavaShape.customOrder(this.shape, axes);
        return new JavaArray(data, shape);
    }

    public JavaArray sum() {
        Boolean[] toCollapse = new Boolean[shape.dimCount];
        fill(toCollapse, TRUE);
        return sum(toCollapse, REMOVE_DIM);
    }

    @Override
    public JavaMaxPool2dResult maxPool2d(int size) {
        JavaShape outShape = (JavaShape) getMaxPool2dResultShape(shape, size);

        TMutableArray tgt = new TMutableArray(new double[outShape.size], outShape);

        int[] inputIndices = shape.newIndexArray();
        int[] tgtIndices = outShape.newIndexArray();
        JavaShape maxIndexShape = createMax2dIndexShape(outShape);
        int[] tmpMaxIndices = maxIndexShape.newIndexArray();
        int[] maxIndexData = new int[maxIndexShape.size];

        fillMax2d(inputIndices, size, tgt, tgtIndices,
                maxIndexShape, maxIndexData, tmpMaxIndices,
                0);

        return new JavaMaxPool2dResult(tgt.migrateToImmutable(), shape, maxIndexShape, maxIndexData);
    }

    private void fillMax2d(int[] inputIndices, int size,
                           TMutableArray tgt, int[] tgtIndices,
                           JavaShape maxIndexShape, int[] maxIndexData, int[] tmpMaxIndices,
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
                               JavaShape maxIndexShape, int[] maxIndexData, int[] tmpMaxIndices) {
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

    @Override
    public NDArray maxPool2dGrad(MaxPool2dResult result) {
        JavaMaxPool2dResult r = (JavaMaxPool2dResult) result;
        return distribute2dMaxGrad(this, r.inputShape, r.maxIndexShape, r.maxIndexData);
    }

    @Override
    public ReluResult relu(double leakyScale) {
        JavaArray copy = normalOrderedCopy();
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

    private static JavaShape createMax2dIndexShape(Shape outShape) {
        int[] idxDims = copyOf(outShape.toDimArray(), outShape.getDimCount() + 1);
        // for (y,x) pair to log the original location of the value in
        // the input matrix.
        idxDims[outShape.getDimCount()] = 2;
        return new JavaShape(idxDims);
    }

    @Override
    public JavaArray sum(Boolean[] dimsToCollapse, DimKeepRemove keepRemove) {
        if (dimsToCollapse.length != shape.dimCount) {
            throw new RuntimeException("input collapse dims must have same length as shape");
        }
        Shape physicalShape = toPhysicalShape(shape, dimsToCollapse);
        int[] dimMapping = createSrcToTargetMapping(dimsToCollapse);

        double[] target = new double[Math.toIntExact(physicalShape.getSize())];
        sum(data, shape, shape.newIndexArray(), 0,
                target, physicalShape, physicalShape.newIndexArray(), dimMapping);

        if (keepRemove == DimKeepRemove.KEEP_DIM) {
            return new JavaArray(target, toPhysicalShapeWithKeep(shape, dimsToCollapse));
        }
        return new JavaArray(target, (JavaShape) physicalShape);
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

    private static int[] createSrcToTargetMapping(Boolean[] dimsToCollapse) {
        int[] mapping = new int[dimsToCollapse.length];
        fill(mapping, -1);
        int idx = 0;
        for (int i = 0; i < dimsToCollapse.length; i++) {
            boolean collapseDimension = (dimsToCollapse[i] != null && dimsToCollapse[i]);
            if (!collapseDimension) {
                mapping[i] = idx;
                idx++;
            }
        }
        return mapping;
    }

    private static Shape toPhysicalShape(Shape shape, Boolean[] dimsToCollapse) {
        int count = countFalse(dimsToCollapse);
        int[] physicalDims = new int[count];
        int idx = 0;
        for (int i = 0; i < dimsToCollapse.length; i++) {
            if (!(dimsToCollapse[i] != null && dimsToCollapse[i])) {
                physicalDims[idx] = shape.at(i);
                idx++;
            }
        }
        return ProviderStore.shape(physicalDims);
    }

    private static JavaShape toPhysicalShapeWithKeep(JavaShape shape, Boolean[] dimsToCollapse) {
        int[] physicalDims = new int[dimsToCollapse.length];
        fill(physicalDims, 1);
        for (int i = 0; i < dimsToCollapse.length; i++) {
            if (!(dimsToCollapse[i] != null && dimsToCollapse[i])) {
                physicalDims[i] = shape.at(i);
            }
        }
        return new JavaShape(physicalDims);
    }

    private static int countFalse(Boolean[] dimsToCollapse) {
        int count = 0;
        for (Boolean c : dimsToCollapse) if (!(c != null && c)) count++;
        return count;
    }

    public JavaArray negate() {
        if (this.shape instanceof ReorderedJavaShape) {
            throw new UnsupportedOperationException("reordered shape not yet supported");
        }
        double[] data = copyOf(this.data, this.data.length);
        for (int i = 0; i < data.length; i++) {
            data[i] *= -1;
        }
        return new JavaArray(data, new JavaShape(this.shape.dims));
    }

    public JavaArray sqr() {
        double[] cp = copyOf(data, data.length);
        for (int i = 0; i < cp.length; i++) {
            cp[i] *= cp[i];
        }

        return new JavaArray(cp, shape.copy());
    }

    public JavaArray pow(double power) {
        double[] data = copyOf(this.data, this.data.length);
        double[] filledData = multiThreadingSupportRun(taskRange(0, this.data.length, 64),
                range -> pow(range.start, range.end, data, power),
                (left, ignored) -> left);

        return new JavaArray(filledData, shape.copy());
    }

    private static double[] pow(int start, int end, double[] data, double power) {
        for (int i = start; i < end; i++) {
            data[i] = Math.pow(data[i], power);
        }
        return data;
    }

    public JavaArray sqrt() {
        double[] data = copyOf(this.data, this.data.length);
        double[] filledData = multiThreadingSupportRun(taskRange(0, data.length, 64),
                range -> sqrt(range.start, range.end, data),
                (left, ignored) -> left);

        return new JavaArray(filledData, shape.copy());
    }

    private static double[] sqrt(int start, int end, double[] data) {
        for (int i = start; i < end; i++) {
            data[i] = Math.sqrt(data[i]);
        }
        return data;
    }

    @Override
    public JavaArray div(NDArray b) {
        return div(this, (JavaArray) b);
    }

    private static JavaArray div(JavaArray a, JavaArray b) {
        if (a.shape.getClass() == b.shape.getClass() &&
                Arrays.equals(a.shape.dims, b.shape.dims)) {
            return fastDiv(a, b);
        }

        validateBroadcastShapes(a.shape, b.shape, -1);
        JavaShape outShape = evalBroadcastOutputShape(a.shape, b.shape);

        int[] indexArray = outShape.newIndexArray();
        double[] data = new double[outShape.size];

        div(a, b,
                data, outShape,
                indexArray, 0);

        return new JavaArray(data, outShape);
    }

    private static JavaArray fastDiv(JavaArray a, JavaArray b) {
        double[] data = copyOf(a.data, a.data.length);
        for (int i = data.length - 1; i >= 0; i--) {
            data[i] /= b.data[i];
        }
        return new JavaArray(data, a.shape.copy());
    }

    private static void div(JavaArray a, JavaArray b,
                            double[] out, JavaShape outShape,
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

    @Override
    public NDArray mul(NDArray other) {
        return mul((JavaArray) other);
    }

    public JavaArray mul(JavaArray b) {
        return mul(this, b);
    }

    private static JavaArray mul(JavaArray a, JavaArray b) {
        if (a.shape.getClass() == b.shape.getClass() &&
                Arrays.equals(a.shape.dims, b.shape.dims)) {
            return fastMul(a, b);
        }

        validateBroadcastShapes(a.shape, b.shape, -1);
        JavaShape outShape = evalBroadcastOutputShape(a.shape, b.shape);

        int[] indexArray = outShape.newIndexArray();
        double[] data = new double[outShape.size];

        mul(a, b,
                data, outShape,
                indexArray, 0);

        return new JavaArray(data, outShape);
    }

    private static JavaArray fastMul(JavaArray a, JavaArray b) {
        double[] data = copyOf(a.data, a.data.length);
        for (int i = data.length - 1; i >= 0; i--) {
            data[i] *= b.data[i];
        }
        return new JavaArray(data, a.shape.copy());
    }

    private static void mul(JavaArray a, JavaArray b,
                            double[] out, JavaShape outShape,
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

    public JavaArray div(double v) {
        double[] data = copyOf(this.data, this.data.length);
        for (int i = 0; i < data.length; i++) {
            data[i] /= v;
        }
        return new JavaArray(data, shape.copy());
    }

    @Override
    public NDArray conv2d(NDArray filter, int offsetY, int offsetX, int outHeight, int outWidth) {
        throw new UnsupportedOperationException();
    }

    public JavaArray mul(double v) {
        double[] data = copyOf(this.data, this.data.length);
        double[] filledData = multiThreadingSupportRun(taskRange(0, data.length, 64),
                range -> mul(range.start, range.end, data, v),
                (left, ignored) -> left);

        return new JavaArray(filledData, shape.copy());
    }

    private static double[] mul(int start, int end, double[] data, double v) {
        for (int i = start; i < end; i++) {
            data[i] *= v;
        }
        return data;
    }

    public JavaArray add(double v) {
        double[] data = copyOf(this.data, this.data.length);
        for (int i = 0; i < data.length; i++) {
            data[i] += v;
        }
        return new JavaArray(data, shape.copy());
    }

    @Override
    public String toString() {
        return StringUtils.toString(this);
    }

    private static class JavaMaxPool2dResult implements MaxPool2dResult {
        private final JavaArray output;
        private final JavaShape inputShape;
        private final JavaShape maxIndexShape;
        private final int[] maxIndexData;

        private JavaMaxPool2dResult(JavaArray output, JavaShape inputShape, JavaShape maxIndexShape, int[] maxIndexData) {
            this.output = output;
            this.inputShape = inputShape;
            this.maxIndexShape = maxIndexShape;
            this.maxIndexData = maxIndexData;
        }

        @Override
        public NDArray getOutput() {
            return output;
        }
    }

    private static class JavaReluResult implements ReluResult {
        private final double[] data;
        private final JavaShape shape;
        private final double[] gradMaskData;

        public JavaReluResult(double[] data, JavaShape shape, double[] gradMaskData) {
            this.data = data;
            this.shape = shape;
            this.gradMaskData = gradMaskData;
        }

        @Override
        public NDArray getOutput() {
            return new JavaArray(data, shape);
        }

        @Override
        public NDArray createMask() {
            return new JavaArray(gradMaskData, shape);
        }
    }

    private static class JavaDropOutResult implements DropOutResult {
        private final JavaArray output;
        private final double[] gradMaskData;
        private final int[] dims;

        public JavaDropOutResult(JavaArray output, double[] gradMaskData, int[] dims) {
            this.output = output;
            this.gradMaskData = gradMaskData;
            this.dims = dims;
        }

        @Override
        public NDArray getOutput() {
            return output;
        }

        @Override
        public NDArray createMask() {
            return new JavaArray(gradMaskData, JavaShape.of(dims));
        }
    }
}
