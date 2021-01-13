package com.codeberry.tadlib.provider;

import com.codeberry.tadlib.array.NDArray;
import com.codeberry.tadlib.array.Shape;

import java.util.Stack;
import java.util.concurrent.Callable;

public class ProviderStore {
    private static Provider provider;

    public static void setProvider(Provider provider) {
        ProviderStore.provider = provider;
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
        return provider.createShape(dims);
    }

    public static NDArray arrayFillWith(Shape shape, double v) {
        return provider.createArrayWithValue(shape, v);
    }

    private static final ThreadLocal<Stack<String>> DEVICE_NAME = ThreadLocal.withInitial(Stack::new);

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
