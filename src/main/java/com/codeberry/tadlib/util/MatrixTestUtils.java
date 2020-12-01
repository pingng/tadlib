package com.codeberry.tadlib.util;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public class MatrixTestUtils {
    public static void assertEqualsMatrix(Object a, Object b) {
        assertEqualsMatrix(a, b, 10e-8);
    }

    public static double assertEqualsMatrix(Object a, Object b, double aspectLimit) {
        double errAspect = calcErrAspect(a, b);
        boolean condition = errAspect < aspectLimit;
        if (!condition) {
            print(a);
            print(b);
        }
        assertTrue(condition, "Err aspect: " + errAspect);
        return errAspect;
    }

    private static void print(Object arr) {
        if (arr.getClass() == double[].class) {
            System.err.println(Arrays.toString((double[]) arr));
        } else {
            System.err.println(Arrays.deepToString((Object[]) arr));
        }
    }

    public static double calcErrAspect(Object a, Object b) {
        assertTrue(a.getClass().isArray());
        assertTrue(b.getClass().isArray());
        assertEquals(dimOf(a), dimOf(b));

        double subSum = sqrSum(a, SumOp.SUB, b);
        double addSum = sqrSum(a, SumOp.ADD, b);

        return Math.sqrt(subSum) / Math.sqrt(addSum);
    }

    private static void assertEquals(Object left, Object right) {
        if (!Objects.equals(left, right)) {
            throw new RuntimeException("assertion failed: not equals\n" +
                    "Left: " + left + "\n" +
                    "Right: " + right);
        }
    }

    private static void assertTrue(boolean condition) {
        assertTrue(condition, "assertion failed");
    }

    private static void assertTrue(boolean condition, String msg) {
        if (!condition) {
            throw new RuntimeException(msg);
        }
    }

    private static double sqrSum(Object a, SumOp op, Object b) {
        if (a.getClass() == double[].class) {
            double[] aArr = (double[]) a;
            double[] bArr = (double[]) b;

            double sum = 0;
            for (int i = 0; i < aArr.length; i++) {
                double aV = aArr[i];
                double bV = bArr[i];

                double r = op.apply(aV, bV);
                sum += r * r;
            }
            return sum;
        } else {
            double sum = 0;
            int len = Array.getLength(a);
            for (int i = 0; i < len; i++) {
                Object aEl = Array.get(a, i);
                Object bEl = Array.get(b, i);
                sum += sqrSum(aEl, op, bEl);
            }
            return sum;
        }
    }

    private enum SumOp {
        ADD {
            @Override
            double apply(double aV, double bV) {
                return aV + bV;
            }
        }, SUB {
            @Override
            double apply(double aV, double bV) {
                return aV - bV;
            }
        };

        abstract double apply(double aV, double bV);
    }

    private static List<Integer> dimOf(Object arr) {
        List<Integer> dims = new ArrayList<>();
        while (arr.getClass().isArray()) {
            dims.add(Array.getLength(arr));
            arr = Array.get(arr, 0);
        }
        return dims;
    }
}
