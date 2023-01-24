package com.codeberry.tadlib.provider.java;

import com.codeberry.tadlib.array.util.MultiDimArrayFlattener;
import com.codeberry.tadlib.provider.Provider;
import com.codeberry.tadlib.util.MultiThreadingSupport;

import java.util.Arrays;

public class JavaProvider implements Provider {

    public JavaProvider() {
        this(
                ThreadMode.MULTI_THREADED
                //ThreadMode.SINGLE_THREADED
        );
    }

    public JavaProvider(ThreadMode mode) {
        if (mode == ThreadMode.MULTI_THREADED) {
            MultiThreadingSupport.enableMultiThreading();
        } else {
            MultiThreadingSupport.disableMultiThreading();
        }
    }

    public enum ThreadMode {
        SINGLE_THREADED, MULTI_THREADED
    }

    @Override
    public NDArray createArray(double v) {
        return new NDArray(v);
    }

    @Override
    public NDArray createArray(Object multiDimArray) {
        MultiDimArrayFlattener<double[]> preparedData = MultiDimArrayFlattener.prepareFlatData(multiDimArray, double[]::new);

        return new NDArray(preparedData.data, new Shape(preparedData.dimensions));
    }

    @Override
    public JavaIntArray createIntArray(Object multiDimArray) {
        MultiDimArrayFlattener<int[]> preparedData = MultiDimArrayFlattener.prepareFlatData(multiDimArray, int[]::new);

        return new JavaIntArray(preparedData.data, new Shape(preparedData.dimensions));
    }

    @Override
    public JavaIntArray createIntArrayWithValue(Shape shape, int v) {
        int[] data = new int[Math.toIntExact(shape.size)];
        Arrays.fill(data, v);
        return new JavaIntArray(data, shape);
    }

    @Override
    public JavaIntArray createIntArray(int v) {
        return new JavaIntArray(v);
    }

    @Override
    public NDArray createArray(double[] data, Shape shape) {
        return new NDArray(data, shape);
    }

    @Override
    public Shape createShape(int... dims) {
        return new Shape(dims);
    }

    @Override
    public NDArray createArrayWithValue(Shape shape, double v) {
        double[] data = new double[Math.toIntExact(shape.size)];
        if (v != 0)
            Arrays.fill(data, v);
        return new NDArray(data, shape);
    }

    @Override
    public String getShortDescription() {
        return "Java";
    }
}
