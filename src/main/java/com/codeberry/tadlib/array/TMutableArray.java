package com.codeberry.tadlib.array;

import java.util.Arrays;

public class TMutableArray {
    private volatile double[] data;
    public final Shape shape;

    public TMutableArray(Shape shape) {
        this(new double[shape.size], shape);
    }

    public TMutableArray(double[] data, Shape shape) {
        this.data = data;
        this.shape = shape;
    }

    public static TMutableArray copyOf(JavaArray src) {
        double[] data = src.getInternalData();
        return new TMutableArray(Arrays.copyOf(data, data.length), src.shape.copy());
    }

    /**
     * @return 0 when out of bounds
     */
    public double dataAt(int... indices) {
        if (shape.isValid(indices)) {
            int idx = shape.calcDataIndex(indices);
            return data[idx];
        }
        return 0;
    }

    public void setAt(int[] indices, double v) {
        int offset = shape.calcDataIndex(indices);
        data[offset] = v;
    }

    /**
     * The current instance cannot be used after this call.
     */
    public synchronized JavaArray migrateToImmutable() {
        JavaArray immutable = new JavaArray(this.data, shape.copy());

        this.data = null;

        return immutable;
    }
}
