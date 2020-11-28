package com.codeberry.tadlib.array;

import org.junit.jupiter.api.Test;

import static java.util.Arrays.deepEquals;
import static java.util.Arrays.deepToString;
import static org.junit.jupiter.api.Assertions.*;

class TArrayMatMul {
    @Test
    public void matmul() {
        TArray a = new TArray(new double[][]{
                {0, 1, 2},
                {3, 4, 5}
        });
        TArray b = new TArray(new double[][]{
                {10},
                {11},
                {13}
        });

        TArray c = a.matmul(b);

        double[][] out = (double[][]) c.toDoubles();
        assertArrayEquals(new double[] {0*10 + 1*11 + 2*13}, out[0]);
        assertArrayEquals(new double[] {3*10 + 4*11 + 5*13}, out[1]);

    }

    @Test
    public void matmul_1D_Right() {
        TArray a = new TArray(new double[][]{
                {1, 2},
                {3, 4}
        });
        TArray b = new TArray(new double[]{10, 20});

        TArray c = a.matmul(b);

        double[] out = (double[]) c.toDoubles();
        assertArrayEquals(new double[] {1*10 + 2*20, 3*10 + 4*20}, out);
    }

    @Test
    public void matmul_1D_Left() {
        TArray a = new TArray(new double[]{10, 20});
        TArray b = new TArray(new double[][]{
                {1, 2},
                {3, 4}
        });

        TArray c = a.matmul(b);

        double[] out = (double[]) c.toDoubles();
        assertArrayEquals(new double[] {1*10 + 3*20, 2*10 + 4*20}, out);
    }

    @Test
    public void matmul_1D_Both() {
        TArray a = new TArray(new double[]{10, 20});
        TArray b = new TArray(new double[]{1, 2});

        TArray c = a.matmul(b);

        double out = (double) c.toDoubles();
        assertEquals(1*10, 2*20, out);
    }

    @Test
    public void matmulInvalidShapes() {
        TArray a = new TArray(new double[]{0, 1, 2, 3, 4, 5});
        TArray b = new TArray(new double[]{1, 10});

        assertDoesNotThrow(() -> a.reshape(1, 3, 2).matmul(b.reshape(2, 1)));
        assertThrows(TArray.InvalidInputShape.class,() ->
                a.reshape(1, 3, 2).matmul(b.reshape(1, 2)));
    }

    @Test
    public void matmulInvalidBroadcast() {
        TArray a = TArray.range(2 * 3 * 2);
        TArray b = TArray.range(3 * 2 * 2);

        assertThrows(TArray.InvalidBroadcastShape.class,
                () -> {
                    TArray c = a.reshape(2, 3, 2)
                            .matmul(b.reshape(3, 2, 2));
                    System.out.println(deepToString((Object[]) c.toDoubles()));
                });
    }

    @Test
    public void matmulBroadcast() {
        TArray a = TArray.range(2 * 3 * 2);
        TArray b = new TArray(new double[]{0.5, 1.0});

        assertDoesNotThrow(() -> {
            TArray _a = a.reshape(6, 2);
            TArray _b = b.reshape(2, 1);
            TArray c = _a.matmul(_b);
            System.out.println("_a = " + toArrStr(_a.toDoubles()));
            System.out.println("_b = " + toArrStr(_b.toDoubles()));
            System.out.println(toArrStr(c.toDoubles()));
            assertTrue(deepEquals(new double[][]
                            {{1.0}, {4.0}, {7.0}, {10.0}, {13.0}, {16.0}}, (Object[]) c.toDoubles()));
        });
        assertDoesNotThrow(() -> {
            TArray _a = a.reshape(2, 3, 2);
            TArray _b = b.reshape(2, 1);
            TArray c = _a.matmul(_b);
            assertTrue(deepEquals(new double[][][]
                            {{{1.0}, {4.0}, {7.0}},
                                    {{10.0}, {13.0}, {16.0}}}, (Object[]) c.toDoubles()));
        });
        assertDoesNotThrow(() -> {
            TArray _a = a.reshape(2, 3, 1, 2);
            TArray _b = b.reshape(2, 1);
            TArray c = _a.matmul(_b);
            assertTrue(deepEquals(new double[][][][]
                            {{{{1.0}}, {{4.0}}, {{7.0}}},
                                    {{{10.0}}, {{13.0}}, {{16.0}}}}, (Object[]) c.toDoubles()));
        });
    }

    static String toArrStr(Object array) {
        return deepToString((Object[]) array)
                .replace("[", "{")
                .replace("]", "}");
    }

    @Test
    public void matmul2() {
        TArray a = new TArray(new double[][]{
                {0, 1, 2},
                {3, 4, 5}
        }).reshape(1, 2, 3);
        TArray b = new TArray(new double[][]{
                {10},
                {11},
                {13}
        });

        TArray c = a.matmul(b);

        Object[] out = (Object[]) c.toDoubles();
        System.out.println(deepToString(out));
    }

}