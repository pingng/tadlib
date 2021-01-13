package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.provider.java.JavaArray;
import com.codeberry.tadlib.provider.opencl.OpenCLProvider;
import com.codeberry.tadlib.util.MatrixTestUtils;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static com.codeberry.tadlib.provider.ProviderStore.*;
import static com.codeberry.tadlib.util.MatrixTestUtils.assertEqualsMatrix;
import static com.codeberry.tadlib.tensor.Ops.div;

public class TensorDivTest {
    @BeforeEach
    public void init() {
//        ProviderStore.setProvider(new JavaProvider()); enableMultiThreading();
        setProvider(new OpenCLProvider());
    }

    @Test
    public void divTest() {
        Tensor dividend = new Tensor(array(new double[]{
                0, 20, 3,
                40, 5, 60,
                7, 80, 9
        }).reshape(1, 3, 3));
        Tensor divisor = new Tensor(array(new double[]{
                1, 3, 7
        }));

        Tensor quotient = div(dividend, divisor);
        quotient.backward(array(new double[]{
                90, 8, 70,
                0.5, 5, -4,
                -50, 2, -1
        }).reshape(1, 3, 3));

        MatrixTestUtils.assertEqualsMatrix(new JavaArray(new double[]{
                        0., 6.6666665, 0.42857143, 40., 1.6666666, 8.571428,
                        7., 26.666666, 1.2857143
                }).reshape(1, 3, 3).toDoubles(),
                quotient.toDoubles());
        System.out.println(Arrays.deepToString((Object[]) dividend.getGradient().toDoubles()));
        assertEqualsMatrix(new JavaArray(new double[]{
                        90., 2.6666667, 10., 0.5, 1.6666666,
                        -0.5714286, -50., 0.6666667, -0.14285715
                }).reshape(1, 3, 3).toDoubles(),
                dividend.getGradient().toDoubles());
        System.out.println(Arrays.toString((double[]) divisor.getGradient().toDoubles()));
        assertEqualsMatrix(new JavaArray(new double[]{
                        330., -38.33333, 0.7959186
                }).toDoubles(),
                divisor.getGradient().toDoubles());

    }
}
