package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.java.NDArray;
import org.junit.jupiter.api.Test;

import static com.codeberry.tadlib.util.MatrixTestUtils.assertEqualsMatrix;

public class TensorReluTest {
    @Test
    public void relu() {
        Tensor input = new Tensor(ProviderStore.array(new double[]{
                0.1, -0.1, 3,
                -4, -5, 9,
                1, 2, -0.0001
        }).reshape(1, 3, 3, 1));

        Tensor relu = Ops.relu(input);
        relu.backward(ProviderStore.array(new double[]{
                10, 20, 30,
                40, 50, 60,
                70, 80, 90
        }).reshape(1, 3, 3, 1));

        NDArray expected = ProviderStore.array(new double[]{
                0.1, 0, 3,
                0, 0, 9,
                1, 2, 0
        }).reshape(1, 3, 3, 1);
        assertEqualsMatrix(expected.toDoubles(), relu.val().toDoubles());

        NDArray expectedGrad = ProviderStore.array(new double[]{
                10, 0, 30,
                0, 0, 60,
                70, 80, 0
        }).reshape(1, 3, 3, 1);
        assertEqualsMatrix(expectedGrad.toDoubles(), input.grad().toDoubles());

    }
}
