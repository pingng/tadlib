package com.codeberry.tadlib.array;

import com.codeberry.tadlib.memorymanagement.DisposalRegister.DisposableContainer;
import com.codeberry.tadlib.provider.ProviderStore;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static com.codeberry.tadlib.array.NDArray.DimKeepRemove.REMOVE_DIM;
import static com.codeberry.tadlib.memorymanagement.DisposalRegister.*;
import static java.lang.Boolean.TRUE;
import static java.lang.Math.min;
import static java.util.Arrays.fill;
import static java.util.Collections.singletonList;

public interface NDArray extends Disposable, DisposableContainer<NDArray> {

    Shape getShape();

    Object toDoubles();

    NDArray normalOrderedCopy();

    /**
     * @deprecated use as little as possible
     */
    double[] getInternalData();

    static void validateConv2dShapes(Shape inputShape, Shape filterShape) {
        if (inputShape.getDimCount() < 4) {
            throw new RuntimeException("input must have 4+ dims");
        }
        if (filterShape.getDimCount() != 4) {
            throw new RuntimeException("filter must have dims [h,w,in,out]");
        }
    }

    double dataAt(int... indices);

    NDArray compare(NDIntArray other, Comparison comparison, double trueValue, double falseValue);

    NDArray compare(NDArray other, Comparison comparison, double trueValue, double falseValue);

    default NDArray sub(NDArray m) {
        return add(m.negate());
    }

    NDArray add(NDArray other);

    NDArray add(double val);

    NDArray mul(NDArray other);

    NDArray div(NDArray other);

    NDArray mul(double val);

    NDArray div(double val);

    default NDArray conv2d(NDArray filter) {
        return conv2d(filter, 0, 0);
    }

    NDArray conv2d(NDArray filter, int offsetY, int offsetX, int outHeight, int outWidth);

    NDArray conv2d(NDArray filter, int offsetY, int offsetX);

    /**
     * @deprecated will be replaced by call to conv2d()
     */
    NDArray calcConv2dFilterGradient(NDArray input, NDArray filter);

    NDArray matmul(NDArray b);

    NDArray transpose(int... axes);

    @Override
    default void prepareDependenciesForDisposal() {
        waitForValueReady();
    }

    default void waitForValueReady() {
        // do nothing
    }

    NDArray negate();

    NDArray sqr();

    NDArray sqrt();

    NDArray rot180(int yAxis, int xAxis);

    NDArray pow(double val);

    NDArray sum(Boolean[] dimsToCollapse, DimKeepRemove keepRemove);

    default NDArray sum() {
        Boolean[] toCollapse = new Boolean[getShape().getDimCount()];
        fill(toCollapse, TRUE);
        return sum(toCollapse, REMOVE_DIM);
    }

    default NDArray sumFirstDims(int firstDimsToRemove, DimKeepRemove keepRemove) {
        Boolean[] dimsToCollapse = new Boolean[getShape().getDimCount()];
        Arrays.fill(dimsToCollapse, false);
        for (int i = 0; i < firstDimsToRemove; i++) {
            dimsToCollapse[i] = true;
        }

        return sum(dimsToCollapse, keepRemove);
    }

    MaxPool2dResult maxPool2d(int size);

    /**
     * @deprecated Replace with more general code
     */
    NDArray maxPool2dGrad(MaxPool2dResult result);

    ReluResult relu(double leakyScale);

    NDArray softmax();

    /**
     * @deprecated Replace with more general ops
     */
    NDArray softMaxGrad(NDArray softmax, NDArray oneHotArray);

    DropOutResult dropOut(Random rnd, double dropoutKeep);

    NDArray withUpdates(List<ValueUpdate> updates);

    @Override
    default List<NDArray> getDisposables() {
        return singletonList(this);
    }

    @Override
    default void dispose() {
        // do nothing
    }

    NDArray clip(Double min, Double max);

    NDArray log();

    NDIntArray argmax(int axis) throws AxisOutOfBounds;

    /**
     * @param indices broadcast not supported
     */
    NDArray getAtIndicesOnAxis(NDIntArray indices, int axis);

    /**
     * @param indices broadcast not supported
     * @param change broadcast not supported
     */
    NDArray withUpdateAtIndicesOnAxis(NDIntArray indices, int axis, NDArray change); // TODO: validate shapes & axis

    enum DimKeepRemove {
        REMOVE_DIM {
            @Override
            public Shape toActualOutShape(Shape inShape, Shape outShapeWithSingleDimensions, Boolean[] dimsToSum) {
                int dimCount = countFalse(dimsToSum);
                int[] dims = new int[dimCount];
                int t = 0;

                int orgCount = inShape.getDimCount();
                for (int i = 0; i < orgCount; i++) {
                    if (!dimsToSum[i]) {
                        dims[t++] = inShape.at(i);
                    }
                }

                return ProviderStore.shape(dims);
            }

            private int countFalse(Boolean[] dimsToSum) {
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

    interface InternalIntReader {
        int readValue(long index);
    }

    interface InternalDoubleReader {
        double readValue(long index);
    }

    NDArray reshape(int... dims);

    default NDArray reshape(Shape shape) {
        if (getShape().getClass() != shape.getClass()) {
            throw new UnsupportedOperationException();
        }
        int[] dims = new int[shape.getDimCount()];
        for (int i = 0; i < dims.length; i++) {
            dims[i] = shape.at(i);
        }
        return reshape(dims);
    }

    interface MaxPool2dResult {
        NDArray getOutput();
    }

    interface ReluResult {
        NDArray getOutput();

        NDArray createMask();
    }

    interface DropOutResult {
        NDArray getOutput();

        NDArray createMask();
    }

    class ValueUpdate {
        public final int offset;
        public final double value;

        public ValueUpdate(int offset, double value) {
            this.offset = offset;
            this.value = value;
        }

        public static ValueUpdate fromIndices(double value, Shape shape, int... indices) {
            return new ValueUpdate(shape.calcDataIndex(indices), value);
        }
    }

    default NDArray subBatch(int batchId, int batchSize) {
        Shape shape = getShape();

        int[] indices = shape.newIndexArray();
        int fromBatchIndex = batchId * batchSize;
        indices[0] = fromBatchIndex;
        int fromOffset = shape.calcDataIndex(indices);
        int endBatchIndex = min(shape.at(0), (batchId + 1) * batchSize);
        indices[0] = endBatchIndex;
        int toOffset = shape.calcDataIndex(indices);

        return subArray(fromBatchIndex, fromOffset, endBatchIndex, toOffset);
    }

    NDArray subArray(int fromBatchIndex, int fromOffset, int endBatchIndex, int toOffset);
}
