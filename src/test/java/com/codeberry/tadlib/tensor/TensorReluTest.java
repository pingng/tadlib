package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.array.TArray;
import org.junit.jupiter.api.Test;

import static com.codeberry.tadlib.tensor.MatrixTestUtils.assertEqualsMatrix;

public class TensorReluTest {
    @Test
    public void relu() {
        Tensor input = new Tensor(new TArray(new double[]{
                0.1, -0.1, 3,
                -4, -5, 9,
                1, 2, -0.0001
        }).reshape(1, 3, 3, 1));

        Tensor relu = Ops.relu(input);
        relu.backward(new TArray(new double[]{
                10, 20, 30,
                40, 50, 60,
                70, 80, 90
        }).reshape(1, 3, 3, 1));

        TArray expected = new TArray(new double[]{
                0.1, 0, 3,
                0, 0, 9,
                1, 2, 0
        }).reshape(1, 3, 3, 1);
        assertEqualsMatrix(expected.toDoubles(), relu.vals.toDoubles());

        TArray expectedGrad = new TArray(new double[]{
                10, 0, 30,
                0, 0, 60,
                70, 80, 0
        }).reshape(1, 3, 3, 1);
        assertEqualsMatrix(expectedGrad.toDoubles(), input.gradient.toDoubles());

    }
}
