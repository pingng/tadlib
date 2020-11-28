package com.codeberry.tadlib.array;

import org.junit.jupiter.api.Test;

import static com.codeberry.tadlib.array.TArrayMatMul.toArrStr;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class TArrayTranspose {
    @Test
    public void transpose2d() {
        TArray a = new TArray(new double[][]{
                {0, 1, 2},
                {3, 4, 5}
        });

        TArray c = a.transpose();

        double[][] out = (double[][]) c.toDoubles();
        System.out.println(toArrStr(out));
        assertArrayEquals(new double[] {0, 3}, out[0]);
        assertArrayEquals(new double[] {1, 4}, out[1]);
        assertArrayEquals(new double[] {2, 5}, out[2]);
    }

    @Test
    public void transposeMatMul() {
        TArray a = new TArray(new double[][]{
                {0, 1},
                {2, 3},
                {4, 5}
        });
        TArray b = new TArray(new double[][]{
                {1, 2},
                {2, 3},
                {3, 4}
        });
        b = b.transpose();

        TArray c = a.matmul(b);

        double[][] out = (double[][]) c.toDoubles();
        System.out.println(toArrStr(a.toDoubles()));
        System.out.println(toArrStr(b.toDoubles()));
        System.out.println(toArrStr(out));
        assertArrayEquals(new double[] {2, 3, 4}, out[0]);
        assertArrayEquals(new double[] {2+6, 4+9, 6+12}, out[1]);
        assertArrayEquals(new double[] {4+10, 8+15, 12+20}, out[2]);
    }

    @Test
    public void transpose3d() {
        TArray a = TArray.range(2*3*2)
                .reshape(2, 3, 2);

        TArray c = a.transpose();

        double[][][] out = (double[][][]) c.toDoubles();
        System.out.println(toArrStr(out));
        assertArrayEquals(new double[] {0, 6}, out[0][0]);
        assertArrayEquals(new double[] {2, 8}, out[0][1]);
        assertArrayEquals(new double[] {4, 10}, out[0][2]);
        assertArrayEquals(new double[] {1, 7}, out[1][0]);
        assertArrayEquals(new double[] {3, 9}, out[1][1]);
        assertArrayEquals(new double[] {5, 11}, out[1][2]);
    }

    @Test
    public void transpose3d_CustomAxis() {
        TArray a = TArray.range(2*3*2)
                .reshape(2, 3, 2);

        TArray c = a.transpose(0, 2, 1);

        double[][][] out = (double[][][]) c.toDoubles();
        System.out.println(toArrStr(out));
        assertArrayEquals(new double[] {0, 2, 4}, out[0][0]);
        assertArrayEquals(new double[] {1, 3, 5}, out[0][1]);
        assertArrayEquals(new double[] {6, 8, 10}, out[1][0]);
        assertArrayEquals(new double[] {7, 9, 11}, out[1][1]);
    }

    @Test
    public void transpose_error() {
        TArray a = TArray.range(2*3*2)
                .reshape(2, 3, 2);

        assertThrows(TArray.DimensionMismatch.class, () ->
                a.transpose(0, 1));
        assertThrows(TArray.DimensionMissing.class, () ->
                a.transpose(0, 0, 2));
    }
}