package com.codeberry.tadlib.array;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static java.util.Arrays.fill;

public abstract class TArrayFactory {
    public static TArray list(double... v) {
        return new TArray(v);
    }

    public static TArray onesShaped(int... dims) {
        return ones(new Shape(dims));
    }

    public static TArray ones(Shape shape) {
        TArray m = zeros(shape);
        fill(m.getInternalData(), 1.0);
        return m;
    }

    public static TArray zerosShaped(int... dims) {
        return zeros(new Shape(dims));
    }

    public static TArray zeros(Shape shape) {
        double[] data = new double[shape.size];
        return new TArray(data, shape.normalOrderedCopy());
    }

    public static TArray value(double v) {
        return new TArray(v);
    }

    public static TArray randMatrixInt(Random rand, int from, int to, int size) {
        return new TArray(randomInt(rand, from, to, size));
    }

    private static double[] randomInt(Random rand, int from, int to, int len) {
        double[] row = new double[len];
        int diff = to - from;
        for (int i = 0; i < row.length; i++) {
            row[i] = from + rand.nextInt(diff);
        }
        return row;
    }

    public static TArray rand(Random rand, int size) {
        return new TArray(random(rand, size));
    }

    public static TArray randWeight(Random rand, Shape shape) {
        return randWeight(rand, shape.size).reshape(shape);
    }

    public static TArray randWeight(Random rand, int size) {
        return new TArray(random(rand, size)).add(-0.5).mul(2.0).mul(Math.sqrt(2./size));
    }

    private static double[] random(Random rand, int len) {
        double[] row = new double[len];
        for (int i = 0; i < row.length; i++) {
            row[i] = rand.nextDouble();
        }
        return row;
    }

    public static TArray range(int count) {
        double[] data = new double[count];
        for (int i = 0; i < count; i++) {
            data[i] = i;
        }
        return new TArray(data);
    }

    public static TArray fillLike(Shape shape, TArray zeroDim) {
        if (zeroDim.shape.dimCount != 0) {
            throw new IllegalArgumentException("value must be of zero dim");
        }
        double v = zeroDim.getInternalData()[0];
        double[] data = new double[shape.size];
        fill(data, v);

        return new TArray(data, shape.copy());
    }

    public static TArray array(double[][] values) {
        NativeArrayConverter preparedData = NativeArrayConverter.prepareData(values);

        return new TArray(preparedData.data, preparedData.shape);
    }

    public static TArray array(double[][][][] values) {
        NativeArrayConverter preparedData = NativeArrayConverter.prepareData(values);

        return new TArray(preparedData.data, preparedData.shape);
    }

    private static class NativeArrayConverter {
        private final double[] data;
        private final Shape shape;

        private NativeArrayConverter(double[] data, Shape shape) {
            this.data = data;
            this.shape = shape;
        }

        private static NativeArrayConverter prepareData(Object array) {
            Shape shape = toShape(array);
            double[] data = shape.toDataArray(array);

            return new NativeArrayConverter(data, shape);
        }

        private static Shape toShape(Object array) {
            List<Integer> dimList = addShape(new ArrayList<>(), array);

            int[] dims = new int[dimList.size()];
            for (int i = 0; i < dimList.size(); i++) {
                dims[i] = dimList.get(i);
            }
            return new Shape(dims);
        }

        private static List<Integer> addShape(ArrayList<Integer> dims, Object array) {
            dims.add(Array.getLength(array));
            if (array.getClass().getComponentType().isArray()) {
                addShape(dims, Array.get(array, 0));
            }
            return dims;
        }
    }
}
