package com.codeberry.tadlib.array;

import com.codeberry.tadlib.util.MultiThreadingSupport;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static com.codeberry.tadlib.array.TArray.DimKeepRemove.REMOVE_DIM;
import static com.codeberry.tadlib.util.MultiThreadingSupport.TaskRange.taskRange;
import static com.codeberry.tadlib.util.MultiThreadingSupport.multiThreadingSupportRun;
import static java.lang.Boolean.TRUE;
import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.util.Arrays.*;

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

    // TODO: remove?
    public void setAt(int[] indices, double v) {
        int offset = shape.calcDataIndex(indices);
        data[offset] = v;
    }

    // TODO: implement as "migrateToImmutable() that nulls the array and hands it to TArray? Avoid array copy.
    public TArray toImmutable() {
        return new TArray(Arrays.copyOf(data, data.length), shape.copy());
    }
}
