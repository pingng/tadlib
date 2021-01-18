package com.codeberry.tadlib.provider.java;

import com.codeberry.tadlib.array.NDIntArray;
import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.array.util.FlatToMultiDimArrayConverter;

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
}
