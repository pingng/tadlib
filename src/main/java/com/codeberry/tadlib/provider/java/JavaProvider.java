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
        return new JavaArray(v);
    }

    @Override
    public NDArray createArray(Object multiDimArray) {
        MultiDimArrayFlattener preparedData = MultiDimArrayFlattener.prepareFlatData(multiDimArray);

        return new JavaArray(preparedData.data, new JavaShape(preparedData.dimensions));
    }

    @Override
    public NDArray createArray(double[] data, Shape shape) {
        return new JavaArray(data, (JavaShape) shape);
    }

    @Override
    public Shape createShape(int... dims) {
        return new JavaShape(dims);
    }

    @Override
    public NDArray createArrayWithValue(Shape shape, double v) {
        double[] data = new double[Math.toIntExact(shape.getSize())];
        Arrays.fill(data, v);
        return new JavaArray(data, (JavaShape) shape);
    }
}
