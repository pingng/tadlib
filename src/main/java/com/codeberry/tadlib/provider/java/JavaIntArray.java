package com.codeberry.tadlib.provider.java;

import com.codeberry.tadlib.array.Comparison;
import com.codeberry.tadlib.array.NDArray;
import com.codeberry.tadlib.array.NDIntArray;
import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.array.util.FlatToMultiDimArrayConverter;

import java.util.function.IntFunction;

public class JavaIntArray implements NDIntArray {
    final int[] data;
    private final Shape shape;

    public JavaIntArray(int[] data, Shape shape) {
        this.data = data;
        this.shape = shape;
    }

    public JavaIntArray(int v) {
        this.data = new int[]{v};
        // zero dim
        this.shape = new JavaShape();
    }

    @Override
    public Object toInts() {
        return FlatToMultiDimArrayConverter.toInts(this.shape, i -> data[(int) i]);
    }

    @Override
    public Shape getShape() {
        return shape;
    }

    /**
     * @return Integer.MIN_VALUE at out of bounds
     */
    @Override
    public int dataAt(int... indices) {
        if (shape.isValid(indices)) {
            int idx = shape.calcDataIndex(indices);
            return data[idx];
        }
        return Integer.MIN_VALUE;
    }

    @Override
    public NDIntArray reshape(int... dims) {
        return new JavaIntArray(data, shape.reshape(dims));
    }

    @Override
    public NDArray compare(NDArray other, Comparison comparison, double trueValue, double falseValue) {
        return other.compare(this, comparison.getFlippedComparison(), trueValue, falseValue);
    }

    @Override
    public NDIntArray compare(NDIntArray other, Comparison comparison, int trueValue, int falseValue) {
        Shape leftShape = this.shape;
        Shape rightShape = other.getShape();
        IntFunction<Integer> left = offset -> this.data[offset];
        IntFunction<Integer> right = offset -> ((JavaIntArray) other).data[offset];

        return CompareHelper.compare(comparison::intIsTrue, trueValue, falseValue,
                leftShape, rightShape, left, right, new IntNDArrayWriter());
    }

    private static class IntNDArrayWriter implements CompareHelper.CompareWriter<Integer, JavaIntArray> {
        private int[] data;

        @Override
        public JavaIntArray scalar(Integer value) {
            return new JavaIntArray(value);
        }

        @Override
        public JavaIntArray toArray(JavaShape shape) {
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
