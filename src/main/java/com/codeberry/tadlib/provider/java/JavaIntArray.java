package com.codeberry.tadlib.provider.java;

import com.codeberry.tadlib.array.Comparison;
import com.codeberry.tadlib.array.util.FlatToMultiDimArrayConverter;
import com.codeberry.tadlib.util.memory.DisposalRegister;

import java.util.function.IntFunction;

public class JavaIntArray implements DisposalRegister.Disposable {
    final int[] data;
    public final Shape shape;

    public JavaIntArray(int[] data, Shape shape) {
        this.data = data;
        this.shape = shape;
    }

    public JavaIntArray(int v) {
        this.data = new int[]{v};
        // zero dim
        this.shape = new Shape();
    }

    public Object toInts() {
        return FlatToMultiDimArrayConverter.toInts(this.shape, i -> data[(int) i]);
    }

    /**
     * @return Integer.MIN_VALUE at out of bounds
     */
    public int dataAt(int... indices) {
        return shape.isValid(indices) ? data[shape.calcDataIndex(indices)] : Integer.MIN_VALUE;
    }

    public JavaIntArray reshape(int... dims) {
        return new JavaIntArray(data, shape.reshape(dims));
    }

    public NDArray compare(NDArray other, Comparison comparison, double trueValue, double falseValue) {
        return other.compare(this, comparison.getFlippedComparison(), trueValue, falseValue);
    }

    public JavaIntArray compare(JavaIntArray other, Comparison comparison, int trueValue, int falseValue) {
        IntFunction<Integer> left = offset -> this.data[offset];
        IntFunction<Integer> right = offset -> other.data[offset];

        return CompareHelper.compare(comparison::intIsTrue, trueValue, falseValue,
                this.shape, other.shape, left, right, new IntNDArrayWriter());
    }

    @Override
    public void prepareDependenciesForDisposal() {
        waitForValueReady();
    }

    public void waitForValueReady() {
        // do nothing
    }

    @Override
    public void dispose() {
        // do nothing
    }

    private static class IntNDArrayWriter implements CompareHelper.CompareWriter<Integer, JavaIntArray> {
        private int[] data;

        @Override
        public JavaIntArray scalar(Integer value) {
            return new JavaIntArray(value);
        }

        @Override
        public JavaIntArray toArray(Shape shape) {
            return new JavaIntArray(data, shape);
        }

        @Override
        public void prepareDate(int size) {
            data = new int[size];
        }

        @Override
        public void write(int offset, Integer outVal) {
            data[offset] = outVal;
        }
    }

}
