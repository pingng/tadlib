package com.codeberry.tadlib.array.util;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MultiDimArrayFlattener {
    public final double[] data;
    public final int[] dimensions;

    private MultiDimArrayFlattener(double[] data, int[] dimensions) {
        this.data = data;
        this.dimensions = dimensions;
    }

    public static MultiDimArrayFlattener prepareFlatData(Object array) {
        int[] dimensions = toDimensions(array);
        double[] data = toDataArray(dimensions, array);

        return new MultiDimArrayFlattener(data, dimensions);
    }

    private static int[] toDimensions(Object array) {
        List<Integer> dimList = addShape(new ArrayList<>(), array);

        int[] dims = new int[dimList.size()];
        for (int i = 0; i < dimList.size(); i++) {
            dims[i] = dimList.get(i);
        }
        return dims;
    }

    private static List<Integer> addShape(ArrayList<Integer> dims, Object array) {
        dims.add(Array.getLength(array));
        if (array.getClass().getComponentType().isArray()) {
            addShape(dims, Array.get(array, 0));
        }
        return dims;
    }

    private static double[] toDataArray(int[] dimensions, Object array) {
        int size = calcSize(dimensions);
        double[] data = new double[size];
        flattenToArray(array, 0, data, 0);
        return data;
    }

    private static int calcSize(int[] dimensions) {
        return Arrays.stream(dimensions)
                .reduce(1, (a, b) -> a * b);
    }

    private static int flattenToArray(Object array, int currentDim, double[] out, int outIndex) {
        int len = Array.getLength(array);
        if (!array.getClass().getComponentType().isArray()) {
            double[] arr = (double[]) array;
            System.arraycopy(arr, 0, out, outIndex, len);
            return outIndex + len;
        } else {
            for (int i = 0; i < len; i++) {
                outIndex = flattenToArray(Array.get(array, i), currentDim+1, out, outIndex);
            }
            return outIndex;
        }
    }

}
