package com.codeberry.tadlib.provider.java;

import java.util.Arrays;

public class JavaShapeRot180 extends JavaShape {
    private final int yAxis;
    private final int xAxis;

    public JavaShapeRot180(JavaShape parent, int yAxis, int xAxis) {
        super(parent.dims);
        this.yAxis = yAxis;
        this.xAxis = xAxis;
    }

    @Override
    public int calcDataIndex(int[] indices) {
        int len = indices.length;
        int[] cp = Arrays.copyOf(indices, len);
        cp[yAxis] = dims[yAxis] - cp[yAxis] - 1;
        cp[xAxis] = dims[xAxis] - cp[xAxis] - 1;
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
