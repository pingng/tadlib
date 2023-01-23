package com.codeberry.tadlib.provider.java;

import com.codeberry.tadlib.array.Shape;

public class ValueUpdate {
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
