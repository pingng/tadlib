package com.codeberry.tadlib.provider;

import com.codeberry.tadlib.array.NDIntArray;
import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.provider.java.JavaShape;
import com.codeberry.tadlib.provider.java.NDArray;
import com.codeberry.tadlib.provider.java.JavaProvider;

import java.util.Stack;
import java.util.concurrent.Callable;

public class ProviderStore {
    private static Provider provider = new JavaProvider();

    public static void setProvider(Provider provider) {
        ProviderStore.provider = provider;
    }

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
        if (dims.length==0)
            return JavaShape.zeroDim;
        return provider.createShape(dims);
    }

    public static NDArray arrayFillWith(Shape shape, double v) {
        return provider.createArrayWithValue(shape, v);
    }
    private static final ThreadLocal<Stack<String>> DEVICE_NAME = ThreadLocal.withInitial(Stack::new);

    public static NDIntArray array(int v) {
        return provider.createIntArray(v);
    }

    public static NDIntArray array(int[] v) {
        return provider.createIntArray(v);
    }

    public static NDIntArray array(int[][] v) {
        return provider.createIntArray(v);
    }

    public static NDIntArray intArrayFillWith(Shape shape, int v) {
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
