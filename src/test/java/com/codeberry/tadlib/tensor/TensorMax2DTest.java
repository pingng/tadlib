package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.java.NDArray;
import com.codeberry.tadlib.provider.java.JavaProvider;
//import com.codeberry.tadlib.provider.opencl.OpenCLProvider;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static com.codeberry.tadlib.util.MatrixTestUtils.assertEqualsMatrix;
import static com.codeberry.tadlib.tensor.Ops.maxpool2d;

public class TensorMax2DTest {
    @BeforeEach
    public void init() {
//        ProviderStore.setProvider(new JavaProvider()); enableMultiThreading();
        ProviderStore.setProvider(new JavaProvider());
    }

    @Test
    public void allOutputsHaveCompleteInputWindow() {
        NDArray m = ProviderStore.array(new double[]{
                0, 10, 2, 20, 8, 30, 4, 80,
                5, 60, 6, 50, 7, 70, 3, 40,
                9, -40, -1, -10, -3, -20, -2, -30,
                -4, 90, -5, -50, -6, -60, -7, -70
        }).reshape(1, 4, 4, 2);
        Tensor input = new Tensor(m);

        Tensor maxed = maxpool2d(input, 2);

        NDArray expected = ProviderStore.array(new double[]{
                6, 60, 8, 80,
                9, 90, -2, -20
        }).reshape(1, 2, 2, 2);

        printMatrix("--- Input", (double[][][][]) input.toDoubles());
        printMatrix("--- Expected", (double[][][][]) expected.toDoubles());
        printMatrix("--- Actual", (double[][][][]) maxed.toDoubles());
        //printMatrix(maxed.m, 2, 2);
        assertEqualsMatrix(expected.toDoubles(),
                maxed.toDoubles());

        NDArray ndArray = maxed.val();
        NDArray g = ProviderStore.array(new double[] {
                1, 2, 3, 4,
                5, 6, 7, 8
        }).reshape(ndArray.shape.toDimArray());
        maxed.backward(g);

        // [[[[0.0, 0.0], [0.0, 0.0], [3.0, 0.0], [0.0, 4.0]],
        //   [[0.0, 2.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        //   [[5.0, 0.0], [0.0, 0.0], [0.0, 8.0], [7.0, 0.0]],
        //   [[0.0, 6.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]]
        NDArray inputGradExpected = ProviderStore.array(new double[]{
                0, 0, 0, 0, 3, 0, 0, 4,
                0, 2, 1, 0, 0, 0, 0, 0,
                5, 0, 0, 0, 0, 8, 7, 0,
                0, 6, 0, 0, 0, 0, 0, 0
        }).reshape(1, 4, 4, 2);
        assertEqualsMatrix(inputGradExpected.toDoubles(),
                input.grad().toDoubles());
    }

    @Test
    public void rightMostAndBottomMostAreMissingInputs() {
        NDArray m = ProviderStore.array(new double[]{
                0, 10, 2,
                9, -40, -1,
                -4, 90, -5
        }).reshape(1, 3, 3, 1);
        Tensor input = new Tensor(m);

        Tensor maxed = maxpool2d(input, 2);

        NDArray expected = ProviderStore.array(new double[]{
                10, 2,
                90, -5
        }).reshape(1, 2, 2, 1);

        printMatrix("--- Input", (double[][][][]) input.toDoubles());
        printMatrix("--- Expected", (double[][][][]) expected.toDoubles());
        printMatrix("--- Actual", (double[][][][]) maxed.toDoubles());
        //printMatrix(maxed.m, 2, 2);
        assertEqualsMatrix(expected.toDoubles(),
                maxed.toDoubles());

        NDArray ndArray = maxed.val();
        NDArray g = ProviderStore.array(new double[] {
                1, 4,
                5, 8
        }).reshape(ndArray.shape.toDimArray());
        maxed.backward(g);

        // [[[[0.0, 0.0], [0.0, 0.0], [3.0, 0.0], [0.0, 4.0]],
        //   [[0.0, 2.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        //   [[5.0, 0.0], [0.0, 0.0], [0.0, 8.0], [7.0, 0.0]],
        //   [[0.0, 6.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]]
        NDArray inputGradExpected = ProviderStore.array(new double[]{
                0, 1, 4,
                0, 0, 0,
                0, 5, 8
        }).reshape(1, 3, 3, 1);
        printMatrix("--- Expected Grad", (double[][][][]) inputGradExpected.toDoubles());
        printMatrix("--- Actual Grad", (double[][][][]) input.grad().toDoubles());

        assertEqualsMatrix(inputGradExpected.toDoubles(),
                input.grad().toDoubles());
    }

    private static void printMatrix(String title, double[][][][] m) {
        System.out.println(title);
        for (double[][][] volume : m) {
            int channels = volume[0][0].length;
            for (int c = 0; c < channels; c++) {
                for (double[][] row : volume) {
                    for (double[] col : row) {
                        System.out.print(col[c]+", ");
                    }
                    System.out.println();
                }
                System.out.println("---");
            }
        }
    }
}
