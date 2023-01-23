package com.codeberry.tadlib.array.util;

import com.codeberry.tadlib.array.*;
import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.java.NDArray;
import com.codeberry.tadlib.provider.java.ReorderedJavaShape;

import java.util.function.Function;

import static java.lang.Math.max;

public abstract class DimensionUtils {

    public static final int SINGLE_DIM_THUS_CAN_BROADCAST = 1;
    public static final int BEYOND_FIRST_DIM = Integer.MIN_VALUE;

    public static void validateBroadcastShapes(Shape a, Shape b, int startDimensionInReverse) {
        if (startDimensionInReverse >= 0) {
            throw new RuntimeException("expects negative start index: " + startDimensionInReverse);
        }
        int max = max(a.getDimCount(), b.getDimCount());
        // NOTE: Reverse loop, aligning comparisons with the last dimensions
        for (int i = startDimensionInReverse; i >= -max; i--) {
            int _a = a.atOrDefault(i, BEYOND_FIRST_DIM);
            int _b = b.atOrDefault(i, BEYOND_FIRST_DIM);
            if (_a == _b ||
                    _a == SINGLE_DIM_THUS_CAN_BROADCAST || _a == BEYOND_FIRST_DIM ||
                    _b == SINGLE_DIM_THUS_CAN_BROADCAST || _b == BEYOND_FIRST_DIM) {
                // ok
            } else {
                throw new InvalidBroadcastShape(a + " vs " + b);
            }
        }
    }

    public static int[] createBroadcastResultDims(Shape a, Shape b) {
        int len = max(a.getDimCount(), b.getDimCount());
        int[] dims = new int[len];
        for (int i = -1; i >= -dims.length; i--) {
            // Either a or b _will_ have a positive value
            dims[len + i] = max(a.atOrDefault(i, Integer.MIN_VALUE),
                    b.atOrDefault(i, Integer.MIN_VALUE));
        }
        return dims;
    }

    public static void validateMatMulShapes(Shape a, Shape b) {
        if (a.at(-1) != b.at(-2)) {
            throw new InvalidInputShape("a: " + a + " b:" + b);
        }
    }

    public static int[] evalMatMulResultDims(Shape a, Shape b) {
        int[] dims = createBroadcastResultDims(a, b);
        dims[dims.length - 2] = a.at(-2);
        dims[dims.length - 1] = b.at(-1);
        return dims;
    }

    public static long mulDimRange(Shape shape, int from, int to) {
        int _f = (from >= 0 ? from : shape.getDimCount() + from);
        int _t = (to >= 0 ? to : shape.getDimCount() + to);
        long count = 1;
        for (int i = _f; i < _t; i++) {
            count *= shape.at(i);
        }
        return count;
    }

    public static long calcExampleCount(Shape shape, ShapeEndType type) {
        return mulDimRange(shape, 0, -type.excludeDims);
    }

    public static int[] calcBroadcastBlockSizes(Shape shape) {
        return calcBroadcastBlockSizes(shape, shape.getDimCount());
    }

    /**
     * @return array of element offsets within each dimension, multiplied by indices
     *         to calculate the offset of a specific coordinate
     */
    public static int[] calcBroadcastBlockSizes(Shape shape, int outDimCount) {
        int[] blockSizes = new int[outDimCount];

        int blockSize = 1;
        for (int i = -1; i >= -shape.getDimCount(); i--) {
            int dimLen = shape.at(i);
            if (dimLen == SINGLE_DIM_THUS_CAN_BROADCAST) {
                //... is a broadcast dim, the final read offset should not
                //    change regardless of what the index is.
                blockSizes[outDimCount + i] = 0;
            } else {
                blockSizes[outDimCount + i] = blockSize;
            }
            blockSize *= dimLen;
        }
        return blockSizes;
    }

    public static int[] calcBlockSizes(Shape shape) {
        return calcBlockSizes(shape, 0, shape.getDimCount());
    }

    public static int[] calcBlockSizes(Shape shape, int from, int to) {
        int dimCount = shape.getDimCount();
        int _f = (from >= 0 ? from : dimCount + from);
        int _t = (to >= 0 ? to : dimCount + to);

        int len = _t - _f;
        if (len > 0) {
            int[] blockSizes = new int[len];
            int blockSize = 1;
            for (int i = blockSizes.length - 1; i >= 0; i--) {
                blockSizes[i] = blockSize;
                blockSize *= shape.at(i + _f);
            }
            return blockSizes;
        }
        return null;
    }

    public static void validateTransposeAxes(Shape shape, int[] axes) {
        if (shape.getDimCount() != axes.length) {
            throw new DimensionMismatch("The requested transpose axes must have equal dimensions: " + shape.getDimCount());
        }
        if (!hasAllAxis(axes)) {
            throw new DimensionMissing("The requested transpose must contain all axis indices: 0-" + (shape.getDimCount() - 1));
        }
    }

    private static boolean hasAllAxis(int[] axes) {
        boolean[] checked = new boolean[axes.length];
        int seen = 0;

        for (int axis : axes) {
            if (!checked[axis]) {
                seen++;
                checked[axis] = true;
            }
        }

        return seen == axes.length;
    }

    public static Shape getMaxPool2dResultShape(Shape inputShape, int size) {
        int[] dims = inputShape.toDimArray();
        int len = dims.length;
        int newH = (dims[len - 3] + size - 1) / size;
        int newW = (dims[len - 2] + size - 1) / size;
        dims[len - 3] = newH;
        dims[len - 2] = newW;

        return ProviderStore.shape(dims);
    }

    public static void validateAxisWithinBounds(Shape shape, int axis) {
        int safeAxis = shape.wrapNegIndex(axis);
        if (safeAxis < 0 || safeAxis >= shape.getDimCount()) {
            throw new AxisOutOfBounds("Valid axes: [0," + (shape.getDimCount() - 1) + "], but is " + axis);
        }
    }

    public static void validateSameDimensionsExcept(String targetName, Shape src, Shape target, int exceptAxis) {
        int safeAxis = src.wrapNegIndex(exceptAxis);
        int srcDimCount = src.getDimCount();
        if (target.getDimCount() != srcDimCount - 1) {
            throw new DimensionMismatch("Expected " + (srcDimCount - 1) + " dimensions: actual=" + target.getDimCount());
        }
        for (int i = 0; i < target.getDimCount(); i++) {
            int srcI = (i < safeAxis ? i : i + 1);
            int expected = src.at(srcI);
            if (expected != target.at(i)) {
                Shape shouldBe = src.removeDimAt(safeAxis);
                throw new DimensionMismatch("Expected " + targetName + ".dimension[" + i + "]==" + expected + ":" +
                        " expectedShape=" + shouldBe + " actualShape=" + target);
            }
        }
    }

    public static int calcOffset(int[] indices, int[] broadcastBlockSizes) {
        int offset = 0;
        for (int i = 0; i < indices.length; i++) {
            offset += indices[i] * broadcastBlockSizes[i];
        }
        return offset;
    }

    public static Shape evalConcatShape(Shape[] shapes, int axis) {
        int[] dims = shapes[0].toDimArray();

        for (int i = 1; i < shapes.length; i++) {
            dims[axis] += shapes[i].at(axis);
        }

        return ProviderStore.shape(dims);
    }

    public static void validateConcatShapes(Shape[] shapes, int axis) {
        Shape first = shapes[0];

        int dimCount = first.getDimCount();
        for (int sI = 1; sI < shapes.length; sI++) {
            Shape s = shapes[sI];
            if (dimCount != s.getDimCount()) {
                throw new DimensionMismatch(dimCount + " != " + s.getDimCount());
            }

            for (int i = 0; i < dimCount; i++) {
                if (i != axis) {
                    if (first.at(i) != s.at(i)) {
                        throw new DimensionMismatch("InputIndex: " + sI + " Dim: " + i + ", " + first.at(i) + " != " + s.at(i));
                    }
                }
            }
        }
    }

    public static Shape evalSplitShape(Shape shape, int axis, int axisLen) {
        int[] dims = shape.toDimArray();
        dims[axis] = axisLen;

        return ProviderStore.shape(dims);
    }

    public static void validateSplitLens(Shape org, int axis, int[] axisLens) {
        int availableLen = org.at(axis);
        int sumSplitLen = 0;
        for (int axisLen : axisLens) {
            sumSplitLen += axisLen;
        }

        if (availableLen != sumSplitLen) {
            throw new DimensionMismatch("Total split axis lengths mismatch: availableLen=" + availableLen + " sumSplitLen=" + sumSplitLen);
        }
    }

    public static Shape[] extractShapes(NDArray[] arrays) {
        Shape[] r = new Shape[arrays.length];
        for (int i = 0; i < arrays.length; i++) {
            r[i] = arrays[i].shape;
        }
        return r;
    }

    public static int[] extractAxisLen(Shape[] shapes, int axis) {
        int[] lens = new int[shapes.length];
        for (int i = 0; i < shapes.length; i++) {
            lens[i] = shapes[i].at(axis);
        }
        return lens;
    }

    public enum ShapeEndType {
        END_WITH__HEIGHT_WIDTH_CHANNEL(3),
        END_WITH__HEIGHT_WIDTH(2);

        private final int excludeDims;

        ShapeEndType(int excludeDims) {
            this.excludeDims = excludeDims;
        }
    }

    /**
     * MatMul requires (at least) 2 dimensions. MatMul with single dimension matrices
     * is possible by expanding the vectors. Follow convention from Numpy.
     */
    public static class MatMulParams {
        public final Shape leftShape;
        public final boolean promoteLeft;
        public final Shape rightShape;
        public final boolean promoteRight;
        private final Function<int[], Shape> factory;

        public MatMulParams(boolean promoteLeft, Shape leftShape,
                            boolean promoteRight, Shape rightShape,
                            Function<int[], Shape> factory) {
            this.promoteLeft = promoteLeft;
            this.promoteRight = promoteRight;
            this.factory = factory;
            this.leftShape = promoteLeft ? prependSingleDim(leftShape) : leftShape;
            this.rightShape = promoteRight ? appendSingleDim(rightShape) : rightShape;
        }

        private Shape prependSingleDim(Shape shape) {
            if (shape instanceof ReorderedJavaShape) {
                throw new UnsupportedOperationException();
            }

            int[] copy = new int[shape.getDimCount() + 1];
            copy(shape, 0, copy, 1);
            copy[0] = 1;
            return factory.apply(copy);
        }

        private static void copy(Shape shape, int srcOffset, int[] copy, int dstOffset) {
            int srcLen = shape.getDimCount();
            int srcIdx = srcOffset;
            for (int i = dstOffset; i < copy.length; i++) {
                if (srcIdx < srcLen) {
                    copy[dstOffset] = shape.at(srcIdx);
                }
                srcIdx++;
            }
        }

        private Shape appendSingleDim(Shape shape) {
            if (shape instanceof ReorderedJavaShape) {
                throw new UnsupportedOperationException();
            }

            int[] copy = new int[shape.getDimCount() + 1];
            copy(shape, 0, copy, 0);
            copy[copy.length - 1] = 1;
            return factory.apply(copy);
        }

        public static MatMulParams expandSingleDimArrays(Shape leftShape, Shape rightShape, Function<int[], Shape> factory) {
            boolean promoteLeft = leftShape.getDimCount() == 1;
            boolean promoteRight = rightShape.getDimCount() == 1;

            return new MatMulParams(promoteLeft, leftShape,
                    promoteRight, rightShape, factory);
        }

        public Shape revertDimExpandOfOutputShape(Shape outShape) {
            Shape finalOutShape = outShape;
            if (promoteLeft) {
                finalOutShape = removeFirstSingleDim(finalOutShape);
            }
            if (promoteRight) {
                finalOutShape = removeLastSingleDim(finalOutShape);
            }
            return finalOutShape;
        }

        private Shape removeFirstSingleDim(Shape shape) {
            int[] copy = new int[shape.getDimCount() - 1];
            copy(shape, 1, copy, 0);

            return factory.apply(copy);
        }

        private Shape removeLastSingleDim(Shape shape) {
            int[] copy = new int[shape.getDimCount() - 1];
            copy(shape, 0, copy, 0);

            return factory.apply(copy);
        }
    }
}
