package com.codeberry.tadlib.array;

import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.opencl.OpenCLProvider;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static com.codeberry.tadlib.array.TArrayFactory.*;
import static java.util.Arrays.deepEquals;
import static java.util.Arrays.deepToString;
import static org.junit.jupiter.api.Assertions.*;

class TArrayMatMul {

    @BeforeEach
    public void init() {
//        ProviderStore.setProvider(new JavaProvider()); enableMultiThreading();
        ProviderStore.setProvider(new OpenCLProvider());
    }

    @Test
    public void matmul() {
        NDArray a = ProviderStore.array(new double[][]{
                {0, 1, 2},
                {3, 4, 5}
        });
        NDArray b = ProviderStore.array(new double[][]{
                {10},
                {11},
                {13}
        });

        NDArray c = a.matmul(b);

        double[][] out = (double[][]) c.toDoubles();
        assertArrayEquals(new double[]{0 * 10 + 1 * 11 + 2 * 13}, out[0]);
        assertArrayEquals(new double[]{3 * 10 + 4 * 11 + 5 * 13}, out[1]);

    }

    @Test
    public void matmulLarge() {
        NDArray a = ProviderStore.arrayFillWith(ProviderStore.shape(40000, 784), 0.0).mul(1);
        NDArray b = ProviderStore.arrayFillWith(ProviderStore.shape(784, 32), 1.0).mul(1);

        NDArray c = a.matmul(b);

        double[][] out = (double[][]) c.toDoubles();
        System.out.println("Rows: "+ out.length);
        System.out.println("Cols: "+ out[0].length);
    }

    @Test
    public void matmul_1D_Right() {
        NDArray a = ProviderStore.array(new double[][]{
                {1, 2},
                {3, 4}
        });
        NDArray b = ProviderStore.array(new double[]{10, 20});

        NDArray c = a.matmul(b);

        double[] out = (double[]) c.toDoubles();
        assertArrayEquals(new double[]{1 * 10 + 2 * 20, 3 * 10 + 4 * 20}, out);
    }

    @Test
    public void matmul_1D_Left() {
        NDArray a = ProviderStore.array(new double[]{10, 20});
        NDArray b = ProviderStore.array(new double[][]{
                {1, 2},
                {3, 4}
        });

        NDArray c = a.matmul(b);

        double[] out = (double[]) c.toDoubles();
        assertArrayEquals(new double[]{1 * 10 + 3 * 20, 2 * 10 + 4 * 20}, out);
    }

    @Test
    public void matmul_1D_Both() {
        NDArray a = ProviderStore.array(new double[]{10, 20});
        NDArray b = ProviderStore.array(new double[]{1, 2});

        NDArray c = a.matmul(b);

        double out = (double) c.toDoubles();
        assertEquals(1 * 10, 2 * 20, out);
    }

    @Test
    public void matmulInvalidShapes() {
        NDArray a = ProviderStore.array(new double[]{0, 1, 2, 3, 4, 5});
        NDArray b = ProviderStore.array(new double[]{1, 10});

        assertDoesNotThrow(() -> a.reshape(1, 3, 2).matmul(b.reshape(2, 1)));
        assertThrows(InvalidInputShape.class, () ->
                a.reshape(1, 3, 2).matmul(b.reshape(1, 2)));
    }

    @Test
    public void matmulInvalidBroadcast() {
        NDArray a = ProviderStore.array(rangeDoubles(2 * 3 * 2));
        NDArray b = ProviderStore.array(rangeDoubles(3 * 2 * 2));

        assertThrows(InvalidBroadcastShape.class,
                () -> {
                    NDArray c = a.reshape(2, 3, 2)
                            .matmul(b.reshape(3, 2, 2));
                    System.out.println(deepToString((Object[]) c.toDoubles()));
                });
    }

    @Test
    public void matmulBig() {
        NDArray a = ProviderStore.array(rangeDoubles(4000 * 512));
        NDArray b = ProviderStore.array(rangeDoubles(512 * 1024));

        long st = System.currentTimeMillis();
        NDArray c = a.reshape(4000, 512)
                .matmul(b.reshape(512, 1024));
        long used = System.currentTimeMillis() - st;
        System.out.println("used = " + used);

        System.out.println(c.getShape());
    }

    @Test
    public void matmulBroadcast() {
        NDArray a = ProviderStore.array(rangeDoubles(2 * 3 * 2));
        NDArray b = ProviderStore.array(new double[]{0.5, 1.0});

        assertDoesNotThrow(() -> {
            NDArray _a = a.reshape(6, 2);
            NDArray _b = b.reshape(2, 1);
            NDArray c = _a.matmul(_b);
            System.out.println("_a = " + toArrStr(_a.toDoubles()));
            System.out.println("_b = " + toArrStr(_b.toDoubles()));
            System.out.println(toArrStr(c.toDoubles()));
            assertTrue(deepEquals(new double[][]
                    {{1.0}, {4.0}, {7.0}, {10.0}, {13.0}, {16.0}}, (Object[]) c.toDoubles()));
        });
        assertDoesNotThrow(() -> {
            NDArray _a = a.reshape(2, 3, 2);
            NDArray _b = b.reshape(2, 1);
            NDArray c = _a.matmul(_b);
            assertTrue(deepEquals(new double[][][]
                    {{{1.0}, {4.0}, {7.0}},
                            {{10.0}, {13.0}, {16.0}}}, (Object[]) c.toDoubles()));
        });
        assertDoesNotThrow(() -> {
            NDArray _a = a.reshape(2, 3, 1, 2);
            NDArray _b = b.reshape(2, 1);
            NDArray c = _a.matmul(_b);
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
        NDArray a = ProviderStore.array(new double[][]{
                {0, 1, 2},
                {3, 4, 5}
        }).reshape(1, 2, 3);
        NDArray b = ProviderStore.array(new double[][]{
                {10},
                {11},
                {13}
        });

        NDArray c = a.matmul(b);

        Object[] out = (Object[]) c.toDoubles();
        System.out.println(deepToString(out));
    }

}