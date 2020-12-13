package com.codeberry.tadlib.array;

import java.util.Arrays;

public class TMutableArray {
    private final double[] data;
    public final Shape shape;

    public TMutableArray(Shape shape) {
        this(new double[shape.size], shape);
    }

    public TMutableArray(double[] data, Shape shape) {
        this.data = data;
        this.shape = shape;
    }

    public static TMutableArray copyOf(TArray src) {
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

    public void addAt(int[] indices, double v) {
        int offset = shape.calcDataIndex(indices);
        data[offset] += v;
    }

    public void setAt(int[] indices, double v) {
        int offset = shape.calcDataIndex(indices);
        data[offset] = v;
    }

    // TODO: implement as "migrateToImmutable() that nulls the array and hands it to TArray? Avoid array copy.
    public TArray toImmutable() {
        return new TArray(Arrays.copyOf(data, data.length), shape.copy());
    }
}
