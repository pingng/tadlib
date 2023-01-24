package com.codeberry.tadlib.array.util;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.IntFunction;

public class MultiDimArrayFlattener<E> {
    public final E data;
    public final int[] dimensions;

    private MultiDimArrayFlattener(E data, int[] dimensions) {
        this.data = data;
        this.dimensions = dimensions;
    }

    public static <E> MultiDimArrayFlattener<E> prepareFlatData(Object array, IntFunction<E> create) {
        int[] dimensions = toDimensions(array);
        E data = toDataArray(dimensions, array, create);

        return new MultiDimArrayFlattener<>(data, dimensions);
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

    private static <E> E toDataArray(int[] dimensions, Object array, IntFunction<E> create) {
        int size = calcSize(dimensions);
        E data = create.apply(size);
        flattenToArray(array, 0, data, 0);
        return data;
    }

    private static int calcSize(int[] dimensions) {
        return Arrays.stream(dimensions)
                .reduce(1, (a, b) -> a * b);
    }

    private static int flattenToArray(Object array, int currentDim, Object out, int outIndex) {
        int len = Array.getLength(array);
        if (!array.getClass().getComponentType().isArray()) {
            System.arraycopy(array, 0, out, outIndex, len);
            return outIndex + len;
        } else {
            for (int i = 0; i < len; i++) {
                outIndex = flattenToArray(Array.get(array, i), currentDim + 1, out, outIndex);
            }
            return outIndex;
        }
    }

}
