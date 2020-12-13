package com.codeberry.tadlib.array;

import com.codeberry.tadlib.util.MultiThreadingSupport;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static com.codeberry.tadlib.array.TArray.DimKeepRemove.REMOVE_DIM;
import static com.codeberry.tadlib.util.MultiThreadingSupport.TaskRange.taskRange;
import static com.codeberry.tadlib.util.MultiThreadingSupport.multiThreadingSupportRun;
import static java.lang.Boolean.TRUE;
import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.util.Arrays.*;

public class TArray {
    public static final TArray ZERO = new TArray(0.0);

    private static final int MAX_STRING_LENGTH = 512;

    private final double[] data;
    public final Shape shape;

    public TArray(double val) {
        this(new double[]{val}, Shape.zeroDim());
    }

    public TArray(double[] data) {
        this(data, new Shape(data.length));
    }

    public TArray(double[] data, Shape shape) {
        this.data = data;
        this.shape = shape;
    }

    public TArray rot180() {
        return new TArray(this.data, new ShapeRot180(this.shape));
    }

    // TODO: remove?
    public void addAt(int[] indices, double v) {
        int offset = shape.calcDataIndex(indices);
        data[offset] += v;
    }

    // TODO: remove?
    public void setAt(int[] indices, double v) {
        int offset = shape.calcDataIndex(indices);
        data[offset] = v;
    }

    public double[] getInternalData() {
        return data;
    }

    public TArray softmax() {
        TArray output = normalOrderedCopy();

        fillSoftMax(output, output.shape.newIndexArray(), 0);

        return output;
    }

    private static void fillSoftMax(TArray tgt, int[] indices, int dim) {
        int len = tgt.shape.at(dim);
        if (indices.length - dim == 1) {
            //_mx = np.max(logits)
            double max = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < len; i++) {
                indices[dim] = i;
                double v = tgt.dataAt(indices);
                if (v > max) {
                    max = v;
                }
            }

            double expSum = 0;
            for (int i = 0; i < len; i++) {
                indices[dim] = i;
                //shifted = logits - _mx
                double shifted = tgt.dataAt(indices) - max;
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
                fillSoftMax(tgt, indices, dim + 1);
            }
        }
    }

    public TArray subBatch(int batchId, int batchSize) {
        TArray src = (shape.getClass() == Shape.class ? this : this.normalOrderedCopy());

        int[] indices = src.shape.newIndexArray();
        int fromBatchIndex = batchId * batchSize;
        indices[0] = fromBatchIndex;
        int fromOffset = src.shape.calcDataIndex(indices);
        int endBatchIndex = min(src.shape.at(0), (batchId + 1) * batchSize);
        indices[0] = endBatchIndex;
        int toOffset = src.shape.calcDataIndex(indices);

        double[] data = new double[toOffset - fromOffset];
        System.arraycopy(src.data, fromOffset, data, 0, data.length);
        int[] dims = src.shape.toDimArray();
        dims[0] = endBatchIndex - fromBatchIndex;
        Shape outShape = new Shape(dims);

        return new TArray(data, outShape);
    }

    public TArray reshape(int... dims) {
        return new TArray(this.data, this.shape.reshape(dims));
    }

    public TArray reshape(Shape shape) {
        return new TArray(this.data, this.shape.reshape(shape.dims));
    }

    public Object toDoubles() {
        if (shape.dimCount == 0) {
            //...no shape, return Double
            return data[0];
        }
        int[] indices = shape.newIndexArray();
        Object arr = shape.newValueArray();
        fillIntoMultiArray(arr, indices, 0);
        return arr;
    }

    private void fillIntoMultiArray(Object arr, int[] indices, int dim) {
        if (indices.length - dim <= 0) {
            throw new RuntimeException("Should not fill into the last index: " +
                    indices.length + "-" + dim);
        }

        int at = shape.at(dim);
        if (indices.length - dim == 1) {
            //...is second last index
            double[] vals = (double[]) arr;
            for (int i = 0; i < vals.length; i++) {
                indices[dim] = i;
                int offset = shape.calcDataIndex(indices);
                vals[i] = data[offset];
            }
        } else {
            for (int i = 0; i < at; i++) {
                Object childArr = Array.get(arr, i);
                indices[dim] = i;
                fillIntoMultiArray(childArr, indices, dim + 1);
            }
        }
    }

    public TArray matmul(TArray b) {
        return matmul(this, b);
    }

    public TArray sub(TArray m) {
        return add(m.negate());
    }

    public TArray add(TArray b) {
        return add(this, b);
    }

    private static TArray add(TArray a, TArray b) {
        if (a.shape.dimCount == 0 &&
                b.shape.dimCount == 0) {
            return new TArray(a.data[0] + b.data[0]);
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

        return new TArray(data, outShape);
    }

    private static TArray fastAdd(TArray a, TArray b) {
        double[] data = copyOf(a.data, a.data.length);
        for (int i = data.length - 1; i >= 0; i--) {
            data[i] += b.data[i];
        }
        return new TArray(data, a.shape.copy());
    }

    public enum Debug {
        NONE(false),
        TRACE(true);
        public final boolean trace;

        Debug(boolean trace) {
            this.trace = trace;
        }
    }

    public TArray conv2d(TArray filter) {
        return conv2d(filter, 0, 0, Debug.NONE);
    }

    public TArray conv2d(TArray filter, int offsetY, int offsetX, Debug debug) {
        return conv2d(this, filter, offsetY, offsetX, debug);
    }

    private static TArray conv2d(TArray input, TArray filter, int offsetY, int offsetX, Debug debug) {
        if (input.shape.dimCount < 4) {
            throw new RuntimeException("input must have 4+ dims");
        }
        if (filter.shape.dimCount != 4) {
            throw new RuntimeException("filter must have dims [h,w,in,out]");
        }

        Shape outShape = evalConv2DShape(input.shape, filter.shape);
        double[] data = new double[outShape.size];

        double[] filledData = multiThreadingSupportRun(taskRange(0, input.shape.at(0)),
                range -> conv2dSegmentedAtFirstDim(range.start, range.end, input, filter, offsetY, offsetX,
                        data, outShape, debug),
                (left, ignored) -> left);

        return new TArray(filledData, outShape);
    }

    private static double[] conv2dSegmentedAtFirstDim(int start, int end,
                                                  TArray input, TArray filter,
                                                  int offsetY, int offsetX,
                                                  double[] data, Shape outShape,
                                                  Debug debug) {
        int[] inIndices = input.shape.newIndexArray();
        int[] fIndices = filter.shape.newIndexArray();
        int[] outIndices = outShape.newIndexArray();
        for (int i = start; i < end; i++) {
            inIndices[0] = i;
            outIndices[0] = i;
            conv2dMain(input, inIndices, 1,
                    filter, fIndices,
                    offsetY, offsetX, data, outShape, outIndices,
                    debug);
        }
        return data;
    }

    private static void conv2dMain(TArray input, int[] inIndices, int inDim,
                                   TArray filter, int[] fIndices, int offsetY, int offsetX,
                                   double[] data, Shape outShape, int[] outIndices,
                                   Debug debug) {
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
                            data, outShape, outIndices,
                            debug);
                }
            }
        } else {
            int len = input.shape.at(inDim);
            for (int i = 0; i < len; i++) {
                inIndices[inDim] = i;
                outIndices[inDim] = i;
                conv2dMain(input, inIndices, inDim +1,
                        filter, fIndices,
                        offsetY, offsetX, data, outShape, outIndices,
                        debug);
            }
        }
    }

    // Eg.: in: <...,5,5,2> filter: <3,3,2,3>
    private static void conv2dAt(TArray input, int[] inIndices,
                                 TArray filter, int[] fIndices, int offsetY, int offsetX,
                                 double[] data, Shape outShape, int[] outIndices,
                                 Debug debug) {
        int fLen = fIndices.length;
        int inLen = inIndices.length;
        int outLen = outIndices.length;
        int inYIdx = inLen - 3;
        int inXIdx = inLen - 2;
        int oldY = inIndices[inYIdx];
        int oldX = inIndices[inXIdx];
        int fH = filter.shape.at(0);
        int fW = filter.shape.at(1);
        int fYOffset = -(fH-1) / 2 + offsetY;
        int fXOffset = -(fW-1) / 2 + offsetX;
        int inCount = input.shape.at(-1);
        int outCount = outShape.at(-1);
        if (debug.trace) {
            System.out.println("=== " + oldX + " x " + oldY);
        }
        for (int outI = 0; outI < outCount; outI++) {
            fIndices[fLen - 1] = outI;
            double v = 0;
            if (debug.trace) {
                System.out.println("out:" + outI);
            }
            for (int inI = 0; inI < inCount; inI++) {
                inIndices[inLen-1] = inI;
                fIndices[fLen-2] = inI;
                if (debug.trace) {
                    System.out.println("in:" + inI);
                }
                for (int y = 0; y < fH; y++) {
                    inIndices[inYIdx] = oldY + y + fYOffset;
                    fIndices[0] = y;
                    for (int x = 0; x < fW; x++) {
                        inIndices[inXIdx] = oldX + x + fXOffset;
                        fIndices[1] = x;
                        double iVal = input.dataAt(inIndices);
                        double fVal = filter.dataAt(fIndices);
                        if (debug.trace) {
                            System.out.print(iVal + "*" + fVal+" ");
                        }
                        v += iVal * fVal;
                    }
                    if (debug.trace) {
                        System.out.println();
                    }
                }
            }
            if (debug.trace) {
                System.out.println("=" + v);
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

    private static Shape evalConv2DShape(Shape input, Shape filter) {
        // todo: handle valid padding
        int[] dims = input.normalOrderedCopy().dims;
        dims[dims.length-1] = filter.at(-1);
        return new Shape(dims);
    }

    private static class MatMulParams {
        private final TArray a;
        private final boolean promoteB;
        private final TArray b;
        private final boolean promoteA;

        public MatMulParams(boolean promoteA, TArray a,
                            boolean promoteB, TArray b) {
            this.promoteA = promoteA;
            this.promoteB = promoteB;
            this.a = promoteA ? a.expandDims(0) : a;
            this.b = promoteB ? b.expandDims(-1) : b;
        }

        static MatMulParams expandSingleDimArrays(TArray a, TArray b) {
            boolean promoteA = a.shape.dimCount == 1;
            boolean promoteB = b.shape.dimCount == 1;

            return new MatMulParams(promoteA, a,
                    promoteB, b);
        }

        Shape revertDimExpandOfOutputShape(Shape outShape) {
            Shape finalOutShape = outShape;
            if (promoteA) {
                finalOutShape = finalOutShape.squeeze(0);
            }
            if (promoteB) {
                finalOutShape = finalOutShape.squeeze(-1);
            }
            return finalOutShape;
        }
    }

    private static TArray matmul(TArray a, TArray b) {
        MatMulParams params = MatMulParams.expandSingleDimArrays(a, b);

        validateMatMulShapes(params.a.shape, params.b.shape);
        validateBroadcastShapes(params.a.shape, params.b.shape, -3);

        Shape outShape = evalMatMulShape(params.a.shape, params.b.shape);
        double[] data = new double[outShape.size];

        int outputRows = outShape.at(-2);
        double[] filledData = multiThreadingSupportRun(
                taskRange(0, outputRows)
                        .withMinimumWorkLength(decideMinimumRowsPerThread(params.a.shape, outShape)),
                range -> matmul(range, params.a, params.b,
                        data, outShape, outShape.newIndexArray(), 0),
                (left, ignored) -> left);

        return new TArray(filledData,
                params.revertDimExpandOfOutputShape(outShape));
    }

    private static int decideMinimumRowsPerThread(Shape leftShape, Shape outShape) {
        int valuesToMulPerOutput = leftShape.at(-1);
        int outputsPerRow = outShape.at(-1);

        return 1 + 512 / (valuesToMulPerOutput * outputsPerRow);
    }

    public TArray normalOrderedCopy() {
        Shape tgtShape = this.shape.normalOrderedCopy();
        double[] data = this.shape.convertDataToShape(this.data, tgtShape);

        return new TArray(data, tgtShape);
    }

    private TArray expandDims(int... indicesForSingleDims) {
        if (shape instanceof ReorderedShape) {
            return expandDims(this.normalOrderedCopy(), indicesForSingleDims);
        }
        return expandDims(this, indicesForSingleDims);
    }

    private static TArray expandDims(TArray m, int... indicesForSingleDims) {
        int[] _tmp = copyOf(indicesForSingleDims, indicesForSingleDims.length);
        for (int i = 0; i < _tmp.length; i++) {
            if(_tmp[i] <= -1)
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

        return new TArray(m.data, new Shape(dimArr));
    }

    private static void validateBroadcastShapes(Shape a, Shape b, int startDimensionInReverse) {
        if (startDimensionInReverse >= 0) {
            throw new RuntimeException("expects negative start index: " + startDimensionInReverse);
        }
        int max = max(a.dimCount, b.dimCount);
        for (int i = startDimensionInReverse; i >= -max; i--) {
            int _a = a.atOrNeg(i);
            int _b = b.atOrNeg(i);
            if (_a == _b || _a == 1 || _a == -1 || _b == 1 || _b == -1) {
                // ok
            } else {
                throw new InvalidBroadcastShape();
            }
        }
    }

    private static void validateMatMulShapes(Shape a, Shape b) {
        if (a.at(-1) != b.at(-2)) {
            throw new InvalidInputShape("a: " + a + " b:" + b);
        }
    }

    private static void add(TArray a, TArray b,
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

    private static double[] matmul(MultiThreadingSupport.TaskRange rowRange,
                               TArray a, TArray b,
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
        int offset = shape.getBroadcastOffset(indices);

        return data[offset];
    }

    private static Shape evalBroadcastOutputShape(Shape a, Shape b) {
        int len = max(a.dimCount, b.dimCount);
        return new Shape(createBroadcastDims(a, b, len));
    }

    private static Shape evalMatMulShape(Shape a, Shape b) {
        int len = max(a.dimCount, b.dimCount);
        int[] dims = createBroadcastDims(a, b, len);
        dims[len - 2] = a.at(-2);
        dims[len - 1] = b.at(-1);
        return new Shape(dims);
    }

    private static int[] createBroadcastDims(Shape a, Shape b, int len) {
        int[] dims = new int[len];
        for (int i = -1; i >= -dims.length; i--) {
            dims[len + i] = max(a.atOrNeg(i), b.atOrNeg(i));
        }
        return dims;
    }

    public TArray transpose(int... axes) {
        ReorderedShape shape = axes.length == 0?
                ReorderedShape.reverseOf(this.shape) : ReorderedShape.customOrder(this.shape, axes);
        return new TArray(data, shape);
    }

    public TArray sum() {
        Boolean[] toCollapse = new Boolean[shape.dimCount];
        fill(toCollapse, TRUE);
        return sumDims(toCollapse, REMOVE_DIM);
    }

    public TArray sumFirstDims(int firstDimsToRemove, DimKeepRemove keepRemove) {
        Boolean[] dimsToCollapse = new Boolean[shape.dimCount];
        for (int i = 0; i < firstDimsToRemove; i++) {
            dimsToCollapse[i] = true;
        }

        return sum(dimsToCollapse, keepRemove);
    }

    private TArray sum(Boolean[] dimsToCollapse, DimKeepRemove keepRemove) {
        if (dimsToCollapse.length != shape.dimCount) {
            throw new RuntimeException("input collapse dims must have same length as shape");
        }
        Shape physicalShape = toPhysicalShape(shape, dimsToCollapse);
        int[] dimMapping = createSrcToTargetMapping(dimsToCollapse);

        double[] target = new double[physicalShape.size];
        sum(data, shape, shape.newIndexArray(), 0,
                target, physicalShape, physicalShape.newIndexArray(), dimMapping);

        if (keepRemove == DimKeepRemove.KEEP_DIM) {
            return new TArray(target, toPhysicalShapeWithKeep(shape, dimsToCollapse));
        }
        return new TArray(target, physicalShape);
    }

    public TArray sumDims(Boolean[] dimsToCollapse, DimKeepRemove keepRemove) {
        return sum(dimsToCollapse, keepRemove);
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
            if (!(dimsToCollapse[i] != null && dimsToCollapse[i])) {
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
        return new Shape(physicalDims);
    }

    private static Shape toPhysicalShapeWithKeep(Shape shape, Boolean[] dimsToCollapse) {
        int[] physicalDims = new int[dimsToCollapse.length];
        fill(physicalDims, 1);
        for (int i = 0; i < dimsToCollapse.length; i++) {
            if (!(dimsToCollapse[i] != null && dimsToCollapse[i])) {
                physicalDims[i] = shape.at(i);
            }
        }
        return new Shape(physicalDims);
    }

    private static int countFalse(Boolean[] dimsToCollapse) {
        int count = 0;
        for (Boolean c : dimsToCollapse) if (!(c != null && c)) count++;
        return count;
    }

    public TArray negate() {
        if (this.shape instanceof ReorderedShape) {
            throw new UnsupportedOperationException("reordered shape not yet supported");
        }
        double[] data = copyOf(this.data, this.data.length);
        for (int i = 0; i < data.length; i++) {
            data[i] *= -1;
        }
        return new TArray(data, new Shape(this.shape.dims));
    }

    public TArray sqr() {
        double[] cp = copyOf(data, data.length);
        for (int i = 0; i < cp.length; i++) {
            cp[i] *= cp[i];
        }

        return new TArray(cp, shape.copy());
    }

    public TArray pow(double power) {
        double[] data = copyOf(this.data, this.data.length);
        double[] filledData = multiThreadingSupportRun(taskRange(0, this.data.length, 64),
                range -> pow(range.start, range.end, data, power),
                (left, ignored) -> left);

        return new TArray(filledData, shape.copy());
    }

    private static double[] pow(int start, int end, double[] data, double power) {
        for (int i = start; i < end; i++) {
            data[i] = Math.pow(data[i], power);
        }
        return data;
    }

    public TArray sqrt() {
        double[] data = copyOf(this.data, this.data.length);
        double[] filledData = multiThreadingSupportRun(taskRange(0, data.length, 64),
                range -> sqrt(range.start, range.end, data),
                (left, ignored) -> left);

        return new TArray(filledData, shape.copy());
    }

    private static double[] sqrt(int start, int end, double[] data) {
        for (int i = start; i < end; i++) {
            data[i] = Math.sqrt(data[i]);
        }
        return data;
    }

    public TArray div(TArray b) {
        return div(this, b);
    }

    private static TArray div(TArray a, TArray b) {
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

        return new TArray(data, outShape);
    }

    private static TArray fastDiv(TArray a, TArray b) {
        double[] data = copyOf(a.data, a.data.length);
        for (int i = data.length - 1; i >= 0; i--) {
            data[i] /= b.data[i];
        }
        return new TArray(data, a.shape.copy());
    }

    private static void div(TArray a, TArray b,
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

    public TArray mul(TArray b) {
        return mul(this, b);
    }

    private static TArray mul(TArray a, TArray b) {
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

        return new TArray(data, outShape);
    }

    private static TArray fastMul(TArray a, TArray b) {
        double[] data = copyOf(a.data, a.data.length);
        for (int i = data.length - 1; i >= 0; i--) {
            data[i] *= b.data[i];
        }
        return new TArray(data, a.shape.copy());
    }

    private static void mul(TArray a, TArray b,
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

    public TArray div(double v) {
        double[] data = copyOf(this.data, this.data.length);
        for (int i = 0; i < data.length; i++) {
            data[i] /= v;
        }
        return new TArray(data, shape.copy());
    }

    public TArray mul(double v) {
        double[] data = copyOf(this.data, this.data.length);
        double[] filledData = multiThreadingSupportRun(taskRange(0, data.length, 64),
                range -> mul(range.start, range.end, data, v),
                (left, ignored) -> left);

        return new TArray(filledData, shape.copy());
    }

    private static double[] mul(int start, int end, double[] data, double v) {
        for (int i = start; i < end; i++) {
            data[i] *= v;
        }
        return data;
    }

    public TArray add(double v) {
        double[] data = copyOf(this.data, this.data.length);
        for (int i = 0; i < data.length; i++) {
            data[i] += v;
        }
        return new TArray(data, shape.copy());
    }

    public enum DimKeepRemove {
        REMOVE_DIM, KEEP_DIM
    }

    public static class InvalidTargetShape extends RuntimeException {
        InvalidTargetShape(String message) {
            super(message);
        }
    }

    public static class InvalidInputShape extends RuntimeException {
        public InvalidInputShape(String msg) {
            super(msg);
        }
    }

    public static class InvalidBroadcastShape extends RuntimeException {
    }

    public static class DimensionMismatch extends RuntimeException {
    }

    public static class DimensionMissing extends RuntimeException {
    }

    public static class CannotSqueezeNoneSingleDimension extends RuntimeException {
        public CannotSqueezeNoneSingleDimension(String msg) {
            super(msg);
        }
    }

    public static class DuplicatedSqueezeDimension extends RuntimeException {
        public DuplicatedSqueezeDimension(String msg) {
            super(msg);
        }
    }

    @Override
    public String toString() {
        StringBuilder buf = new StringBuilder(shape.toString()).append(" ");
        Object o = toDoubles();
        if (o instanceof Double) {
            buf.append((double) o);
        } else if (o instanceof double[]) {
            buf.append(Arrays.toString((double[]) o));
        } else {
            buf.append(deepToString((Object[]) o));
        }
        if (buf.length() > MAX_STRING_LENGTH - 3) {
            buf.setLength(MAX_STRING_LENGTH);
            buf.append("...");
        }
        return buf.toString();
    }

}
