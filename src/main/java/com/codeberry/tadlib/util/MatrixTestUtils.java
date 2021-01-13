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

    public static double assertEqualsMatrix(Object expected, Object actual, double aspectLimit) {
        double errAspect = calcErrAspect(expected, actual);
        boolean condition = errAspect < aspectLimit;
        if (!condition) {
            System.err.println("Expected");
            print(expected);
            System.err.println("Actual");
            print(actual);
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

        // Exception for cases when a or b is very small or is zero
        if (subSum < 3.5e-21 && subSum < 3.5e-21) {
            return subSum;
        }

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
            throw new AssertionError(msg);
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
            int lenA = Array.getLength(a);
            int lenB = Array.getLength(b);
            assertEquals(lenA, lenB);
            for (int i = 0; i < lenA; i++) {
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
