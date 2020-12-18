package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.array.JavaArray;
import org.junit.jupiter.api.Test;

import static com.codeberry.tadlib.util.MatrixTestUtils.assertEqualsMatrix;
import static com.codeberry.tadlib.tensor.Ops.maxpool2d;

public class TensorMax2DTest {
    @Test
    public void testMethod() {
        JavaArray m = new JavaArray(new double[]{
                0, 10, 2, 20, 8, 30, 4, 80,
                5, 60, 6, 50, 7, 70, 3, 40,
                9, -40, -1, -10, -3, -20, -2, -30,
                -4, 90, -5, -50, -6, -60, -7, -70
        }).reshape(1, 4, 4, 2);
        Tensor input = new Tensor(m);

        //printMatrix(m);
        Tensor maxed = maxpool2d(input, 2);
        JavaArray g = new JavaArray(new double[] {
                1, 2, 3, 4,
                5, 6, 7, 8
        }).reshape(maxed.vals.shape.toDimArray());
        maxed.backward(g);

        // [[[[0.0, 0.0], [0.0, 0.0], [3.0, 0.0], [0.0, 4.0]],
        //   [[0.0, 2.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        //   [[5.0, 0.0], [0.0, 0.0], [0.0, 8.0], [7.0, 0.0]],
        //   [[0.0, 6.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]]
        JavaArray inputGradExpected = new JavaArray(new double[]{
                0, 0, 0, 0, 3, 0, 0, 4,
                0, 2, 1, 0, 0, 0, 0, 0,
                5, 0, 0, 0, 0, 8, 7, 0,
                0, 6, 0, 0, 0, 0, 0, 0
        }).reshape(1, 4, 4, 2);
        assertEqualsMatrix(inputGradExpected.toDoubles(),
                input.gradient.toDoubles());

        JavaArray expected = new JavaArray(new double[]{
                6, 60, 8, 80,
                9, 90, -2, -20
        }).reshape(1, 2, 2, 2);
        //printMatrix(maxed.m, 2, 2);
        assertEqualsMatrix(expected.toDoubles(),
                maxed.toDoubles());
    }

    private static void printMatrix(JavaArray m, int h, int w) {
        int[] indices = m.shape.newIndexArray();
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                indices[1] = y;
                indices[2] = x;
                indices[3] = 1;
                System.out.print(m.dataAt(indices)+" ");
            }
            System.out.println();
        }
    }
}
