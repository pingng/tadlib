package com.codeberry.tadlib.tensor.conv2ddata;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;

public class Conv2DExample {
    public Config config;
    public Object input;
    public Object filter;
    public Object y;
    public Object grad_y;
    public Object grad_input;
    public Object grad_filter;

    public void convertListsToArrays() {
        input = convert(input);
        filter = convert(filter);
        y = convert(y);
        grad_y = convert(grad_y);
        grad_input = convert(grad_input);
        grad_filter = convert(grad_filter);
    }

    private static Object convert(Object input) {
        int[] dims = getArrayDims(input);
        Object data = Array.newInstance(double.class, dims);
        List<?> root = (List<?>) input;
        fillInto(root, data, dims.length, 0);
        return data;
    }

    private static void fillInto(List<?> l, Object data, int maxDim, int dim) {
        if (maxDim - dim == 1) {
            List<Double> list = (List<Double>) l;
            double[] tgt = (double[]) data;
            for (int i = 0; i < list.size(); i++) {
                tgt[i] = list.get(i);
            }
        } else {
            for (int i = 0; i < l.size(); i++) {
                Object o = l.get(i);
                Object subData = Array.get(data, i);
                fillInto((List<?>) o, subData, maxDim, dim + 1);
            }
        }
    }

    private static int[] getArrayDims(Object input) {
        List<Integer> dims = new ArrayList<>();
        Object el = input;
        while (List.class.isAssignableFrom(el.getClass())) {
            List<?> list = (List<?>) el;
            dims.add(list.size());
            el = list.get(0);
        }
        int[] dimsArr = new int[dims.size()];
        for (int i = 0; i < dims.size(); i++) {
            dimsArr[i] = dims.get(i);
        }
        return dimsArr;
    }

    public static class Config {
        public String name;
        public int input_size;
        public int filter_size;
        public int input_channels;
        public int output_channels;
    }

}
