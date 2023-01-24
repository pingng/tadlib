package com.codeberry.tadlib.tensor;

import org.junit.jupiter.api.Test;

import static com.codeberry.tadlib.provider.ProviderStore.array;
import static com.codeberry.tadlib.util.MatrixTestUtils.assertEqualsMatrix;

public class TensorSumTest {

    @Test
    public void sum() {
        Tensor x = new Tensor(new double[]{
                0.01, 0.1, -1,
                1, 10, -2,
                0.02, 0.2, -3,
                2, 20, -4,
                0.03, 0.3, -5,
                3, 30, -6,
                0.04, 0.4, -7,
                4, 40, -8
        }, 2, 2, 2, 3);

        Tensor y = Ops.SUM(x, 0, 1, 2);

        var Y = y.val();
        //assertEquals(_y.value(), Y);
        y.backward(array(new double[]{0.5, 5.5, 10.9}));

        assertEqualsMatrix(array(new double[]{
                        (0.01 + 1 + 0.02 + 2 + 0.03 + 3 + 0.04 + 4),
                        (0.1 + 10 + 0.2 + 20 + 0.3 + 30 + 0.4 + 40),
                        (-1.0 - 2 - 3 - 4 - 5 - 6 - 7 - 8)
                }).toDoubles(),
                Y.toDoubles());

        assertEqualsMatrix(array(new double[]{
                        0.5, 5.5, 10.9,
                        0.5, 5.5, 10.9,
                        0.5, 5.5, 10.9,
                        0.5, 5.5, 10.9,
                        0.5, 5.5, 10.9,
                        0.5, 5.5, 10.9,
                        0.5, 5.5, 10.9,
                        0.5, 5.5, 10.9
                }).reshape(2, 2, 2, 3).toDoubles(),
                x.grad().toDoubles());
    }

    @Test
    public void sum2() {
        Tensor input = new Tensor(array(new double[]{
                0.01, 0.1, -1,
                1, 10, -2,

                0.02, 0.2, -3,
                2, 20, -4,

                0.03, 0.3, -5,
                3, 30, -6,

                0.04, 0.4, -7,
                4, 40, -8
        }).reshape(2, 2, 2, 3));

        Tensor sum = Ops.sum(input, 0, 2, 3);
        sum.backward(array(new double[]{10, 20}));

        assertEqualsMatrix(array(new double[]{
                        (0.01 + 0.1 + -1 +
                                1 + 10 + -2 +
                                0.03 + 0.3 + -5 +
                                3 + 30 + -6),

                        (0.02 + 0.2 + -3 +
                                2 + 20 + -4 +
                                0.04 + 0.4 + -7 +
                                4 + 40 + -8)
                }).toDoubles(),
                sum.val().toDoubles());
        assertEqualsMatrix(array(new double[]{
                        10.0, 10.0, 10.0,
                        10.0, 10.0, 10.0,

                        20.0, 20.0, 20.0,
                        20.0, 20.0, 20.0,

                        10.0, 10.0, 10.0,
                        10.0, 10.0, 10.0,

                        20.0, 20.0, 20.0,
                        20.0, 20.0, 20.0
                }).reshape(2, 2, 2, 3).toDoubles(),
                input.grad().toDoubles());

    }
}
