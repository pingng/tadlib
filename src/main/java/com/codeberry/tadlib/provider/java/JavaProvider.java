package com.codeberry.tadlib.provider.java;

import com.codeberry.tadlib.array.*;
import com.codeberry.tadlib.array.util.MultiDimArrayFlattener;
import com.codeberry.tadlib.provider.Provider;
import com.codeberry.tadlib.util.MultiThreadingSupport;

import java.util.Arrays;

public class JavaProvider implements Provider {

    public JavaProvider() {
        this(ThreadMode.MULTI_THREADED);
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

        return new NDArray(preparedData.data, new JavaShape(preparedData.dimensions));
    }

    @Override
    public NDIntArray createIntArray(Object multiDimArray) {
        MultiDimArrayFlattener<int[]> preparedData = MultiDimArrayFlattener.prepareFlatData(multiDimArray, int[]::new);

        return new JavaIntArray(preparedData.data, new JavaShape(preparedData.dimensions));
    }

    @Override
    public NDIntArray createIntArrayWithValue(Shape shape, int v) {
        int[] data = new int[Math.toIntExact(shape.getSize())];
        Arrays.fill(data, v);
        return new JavaIntArray(data, shape);
    }

    @Override
    public NDIntArray createIntArray(int v) {
        return new JavaIntArray(v);
    }

    @Override
    public NDArray createArray(double[] data, Shape shape) {
        return new NDArray(data, (JavaShape) shape);
    }

    @Override
    public Shape createShape(int... dims) {
        return new JavaShape(dims);
    }

    @Override
    public NDArray createArrayWithValue(Shape shape, double v) {
        double[] data = new double[Math.toIntExact(shape.getSize())];
        Arrays.fill(data, v);
        return new NDArray(data, (JavaShape) shape);
    }

    @Override
    public String getShortDescription() {
        return "Java";
    }
}
