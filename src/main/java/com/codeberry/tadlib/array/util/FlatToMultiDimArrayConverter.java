package com.codeberry.tadlib.array.util;

import com.codeberry.tadlib.provider.java.NDArray;
import com.codeberry.tadlib.provider.java.Shape;

import java.lang.reflect.Array;

public abstract class FlatToMultiDimArrayConverter {
    public static Object toInts(Shape shape, NDArray.InternalIntReader r) {
        if (shape.getDimCount() == 0) {
            //...no shape, return Double
            return r.readValue(0);
        }
        int[] indices = shape.newIndexArray();
        Object arr = shape.newValueArray(int.class);
        fillIntoMultiIntArray(shape, r,
                arr, indices, 0);
        return arr;
    }

    private static void fillIntoMultiIntArray(Shape shape, NDArray.InternalIntReader r, Object arr, int[] indices, int dim) {
        if (indices.length - dim <= 0) {
            throw new RuntimeException("Should not fill into the last index: " +
                    indices.length + "-" + dim);
        }

        int at = shape.at(dim);
        if (indices.length - dim == 1) {
            //...is second last index
            int[] vals = (int[]) arr;
            for (int i = 0; i < vals.length; i++) {
                indices[dim] = i;
                int offset = shape.calcDataIndex(indices);
                vals[i] = r.readValue(offset);
            }
        } else {
            for (int i = 0; i < at; i++) {
                Object childArr = Array.get(arr, i);
                indices[dim] = i;
                fillIntoMultiIntArray(shape, r, childArr, indices, dim + 1);
            }
        }
    }

    public static Object toDoubles(Shape shape, NDArray.InternalDoubleReader r) {
        if (shape.dimCount == 0) {
            //...no shape, return Double
            return r.readValue(0);
        }
        int[] indices = shape.newIndexArray();
        Object arr = shape.newValueArray(double.class);
        fillIntoMultiDoubleArray(shape, r,
                arr, indices, 0);
        return arr;
    }

    private static void fillIntoMultiDoubleArray(Shape shape, NDArray.InternalDoubleReader r, Object arr, int[] indices, int dim) {
        if (indices.length - dim <= 0) {
            throw new RuntimeException("Should not fill into the last index: " +
                    indices.length + "-" + dim);
        }

        int at = shape.at(dim);
        if (indices.length - dim == 1) {
            //...is second last index
            double[] vals = (double[]) arr;
            for (int i = 0; i < vals.length; i++) {
                indices[dim] = i;
                int offset = shape.calcDataIndex(indices);
                vals[i] = r.readValue(offset);
            }
        } else {
            for (int i = 0; i < at; i++) {
                Object childArr = Array.get(arr, i);
                indices[dim] = i;
                fillIntoMultiDoubleArray(shape, r, childArr, indices, dim + 1);
            }
        }
    }
}
