package com.codeberry.tadlib.array;

import java.lang.reflect.Array;

import static java.lang.Math.min;
import static java.util.Arrays.copyOf;

public class ReorderedShape extends Shape {
    private final int[] order;
    private final int[] blockSizes;

    public ReorderedShape(int[] dims, int[] order) {
        super(dims);
        this.order = order;
        blockSizes = new int[dims.length];

        int blockSize = 1;
        for (int i = blockSizes.length - 1; i >= 0; i--) {
            blockSizes[i] = blockSize;
            blockSize *= dims[i];
        }
    }

    @Override
    public int at(int dim) {
        return super.at(order[wrapIndex(dim)]);
    }

    @Override
    public int atOrNeg(int dim) {
        return super.atOrNeg(order[wrapIndex(dim)]);
    }

    @Override
    public Object newValueArray() {
        int[] d = new int[dimCount];
        for (int i = 0; i < d.length; i++) {
            d[i] = dims[order[i]];
        }
        return Array.newInstance(double.class, d);
    }

    @Override
    public int calcDataIndex(int[] indices) {
        int idx = 0;
        for (int i = indices.length - 1; i >= 0; i--) {
            int orgIndex = order[i];
            idx += indices[i] * blockSizes[orgIndex];
        }
        return idx;
    }

    @Override
    int getBroadcastOffset(int[] indices) {
        int offset = 0;
        int dimCount = this.dimCount;
        for (int i = -1; i >= -dimCount; i--) {
            int shapeDimSize = at(i);
            int index = min(indices[indices.length + i], shapeDimSize - 1);
            offset += blockSizes[order[dimCount + i]] * index;
        }
        return offset;
    }

    @Override
    public Shape normalOrderedCopy() {
        int[] dims = new int[dimCount];
        for (int i = 0; i < dims.length; i++) {
            dims[i] = at(i);
        }
        return new Shape(dims);
    }

    @Override
    public Shape copy() {
        return new ReorderedShape(copyOf(dims, dimCount), copyOf(order,dimCount));
    }

    @Override
    public boolean isValid(int[] indices) {
        for (int i = indices.length - 1; i >= 0; i--) {
            int index = indices[i];
            if (index <= -1 || index >= at(i))
                return false;
        }
        return true;
    }

    public static ReorderedShape reverseOf(Shape shape) {
        int[] order = new int[shape.dimCount];
        for (int i = 0; i < order.length; i++) {
            order[i] = order.length - 1 - i;
        }
        return new ReorderedShape(shape.dims, order);
    }

    public static ReorderedShape customOrder(Shape shape, int[] axes) {
        if (shape.dimCount != axes.length) {
            throw new TArray.DimensionMismatch();
        }
        if (!hasAllAxis(axes)) {
            throw new TArray.DimensionMissing();
        }
        return new ReorderedShape(shape.dims, axes);
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
}
