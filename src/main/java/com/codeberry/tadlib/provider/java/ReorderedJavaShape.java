package com.codeberry.tadlib.provider.java;

import static com.codeberry.tadlib.array.util.DimensionUtils.validateTransposeAxes;
import static java.lang.Math.min;
import static java.util.Arrays.copyOf;

public class ReorderedJavaShape extends Shape {
    private final int[] order;
    private final int[] blockSizes;

    public ReorderedJavaShape(int[] dims, int[] order) {
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

//    @Override
//    public Object newValueArray() {
//        int[] d = new int[dimCount];
//        for (int i = 0; i < d.length; i++) {
//            d[i] = dims[order[i]];
//        }
//        return Array.newInstance(double.class, d);
//    }

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
        return new ReorderedJavaShape(copyOf(dims, dimCount), copyOf(order, dimCount));
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

    public static ReorderedJavaShape reverseOf(Shape shape) {
        int[] order = new int[shape.dimCount];
        for (int i = 0; i < order.length; i++) {
            order[i] = order.length - 1 - i;
        }
        return new ReorderedJavaShape(shape.dims, order);
    }

    public static ReorderedJavaShape customOrder(Shape shape, int[] axes) {
        validateTransposeAxes(shape, axes);

        return new ReorderedJavaShape(shape.dims, axes);
    }

}
