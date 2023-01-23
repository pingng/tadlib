package com.codeberry.tadlib.provider.java;

import com.codeberry.tadlib.array.Shape;

public class TMutableArray {
    private volatile double[] data;
    public final Shape shape;

    public TMutableArray(JavaShape shape) {
        this(new double[shape.size], shape);
    }

    public TMutableArray(double[] data, Shape shape) {
        this.data = data;
        this.shape = shape;
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
        setAtOffset(offset, v);
    }

    public void setAtOffset(int offset, double v) {
        data[offset] = v;
    }

    public double[] getData() {
        return data;
    }

    /**
     * The current instance cannot be used after this call.
     */
    public synchronized NDArray migrateToImmutable() {
        NDArray immutable = new NDArray(this.data, (JavaShape) shape);

        this.data = null;

        return immutable;
    }
}
