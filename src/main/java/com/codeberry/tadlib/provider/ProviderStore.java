package com.codeberry.tadlib.provider;

import com.codeberry.tadlib.provider.java.JavaIntArray;
import com.codeberry.tadlib.provider.java.JavaProvider;
import com.codeberry.tadlib.provider.java.NDArray;
import com.codeberry.tadlib.provider.java.Shape;

import java.util.Stack;
import java.util.concurrent.Callable;

public class ProviderStore {
    private static final Provider provider = new JavaProvider();

    public static String getProviderShortDescription() {
        return provider.getShortDescription();
    }

    public static NDArray array(double v) {
        return provider.createArray(v);
    }

    public static NDArray array(double[] v) {
        return provider.createArray(v);
    }

    public static NDArray array(double[][] v) {
        return provider.createArray(v);
    }

    public static NDArray array(double[][][][] v) {
        return provider.createArray(v);
    }

    public static NDArray array(double[] data, Shape shape) {
        return provider.createArray(data, shape);
    }

    public static Shape shape(int... dims) {
        return dims.length == 0 ? Shape.zeroDim : provider.createShape(dims);
    }

    public static NDArray arrayFillWith(Shape shape, double v) {
        return provider.createArrayWithValue(shape, v);
    }

    private static final ThreadLocal<Stack<String>> DEVICE_NAME = ThreadLocal.withInitial(Stack::new);

    public static JavaIntArray array(int v) {
        return provider.createIntArray(v);
    }

    public static JavaIntArray array(int[] v) {
        return provider.createIntArray(v);
    }

    public static JavaIntArray array(int[][] v) {
        return provider.createIntArray(v);
    }

    public static JavaIntArray intArrayFillWith(Shape shape, int v) {
        return provider.createIntArrayWithValue(shape, v);
    }

    public static String getDeviceName() {
        Stack<String> nameStack = DEVICE_NAME.get();
        if (nameStack.isEmpty()) {
            return null;
        }
        return nameStack.peek();
    }

    public static <R> R onDevice(String name, Callable<R> r) {
        Stack<String> nameStack = DEVICE_NAME.get();
        try {
            nameStack.push(name);
            try {
                return r.call();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        } finally {
            nameStack.pop();
        }
    }
}
