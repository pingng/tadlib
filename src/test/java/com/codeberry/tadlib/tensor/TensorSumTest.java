package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.provider.opencl.OpenCLProvider;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static com.codeberry.tadlib.provider.ProviderStore.array;
import static com.codeberry.tadlib.provider.ProviderStore.setProvider;
import static com.codeberry.tadlib.util.MatrixTestUtils.assertEqualsMatrix;

public class TensorSumTest {
    @BeforeEach
    public void init() {
//        ProviderStore.setProvider(new JavaProvider());
        setProvider(new OpenCLProvider());
    }

    @Test
    public void sum() {
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

        Tensor sum = Ops.sum(input, 0, 1, 2);
        sum.backward(array(new double[]{0.5, 5.5, 10.9}));

        assertEqualsMatrix(array(new double[]{
                        (0.01 + 1 + 0.02 + 2 + 0.03 + 3 + 0.04 + 4),
                        (0.1 + 10 + 0.2 + 20 + 0.3 + 30 + 0.4 + 40),
                        (-1. - 2 - 3 - 4 - 5 - 6 - 7 - 8)
                }).toDoubles(),
                sum.getVals().toDoubles());
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
                input.getGradient().toDoubles());
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
                sum.getVals().toDoubles());
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
                input.getGradient().toDoubles());

    }
}
