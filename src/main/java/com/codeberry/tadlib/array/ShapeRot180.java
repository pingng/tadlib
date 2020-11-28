package com.codeberry.tadlib.array;

import java.util.Arrays;

public class ShapeRot180 extends Shape {
    public ShapeRot180(Shape parent) {
        super(parent.dims);
    }

    @Override
    public int calcDataIndex(int[] indices) {
        int len = indices.length;
        int[] cp = Arrays.copyOf(indices, len);
        cp[len - 4] = dims[len-4] - cp[len - 4] - 1;
        cp[len - 3] = dims[len-3] - cp[len - 3] - 1;
        return super.calcDataIndex(cp);
    }

    // Rot180 has the same dimensions, so we use the super implementation
    @Override
    public boolean isValid(int[] indices) {
        return super.isValid(indices);
    }

    @Override
    int getBroadcastOffset(int[] indices) {
        throw new UnsupportedOperationException();
    }
}
