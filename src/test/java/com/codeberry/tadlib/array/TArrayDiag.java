package com.codeberry.tadlib.array;

import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.java.JavaProvider;
import com.codeberry.tadlib.provider.opencl.OpenCLProvider;
import com.codeberry.tadlib.util.MatrixTestUtils;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static com.codeberry.tadlib.provider.ProviderStore.array;

public class TArrayDiag {

    @BeforeEach
    public void init() {
//        ProviderStore.setProvider(new JavaProvider());
        ProviderStore.setProvider(new OpenCLProvider());
    }

    @Test
    public void test1D() {
        NDArray input = array(new double[]{
                1, 2, 3, 4, 5, 6
        })
                .reshape(6);

        NDArray diagonal = input.diag();

        NDArray expected = array(new double[]{
                1, 0, 0, 0, 0, 0,
                0, 2, 0, 0, 0, 0,
                0, 0, 3, 0, 0, 0,
                0, 0, 0, 4, 0, 0,
                0, 0, 0, 0, 5, 0,
                0, 0, 0, 0, 0, 6
        }).reshape(6, 6);

        MatrixTestUtils.assertEqualsMatrix(expected.toDoubles(), diagonal.toDoubles());
    }

    @Test
    public void test2D() {
        NDArray input = array(new double[]{
                1, 2, 3,
                4, 5, 6
        })
                .reshape(2, 3);

        NDArray diagonal = input.diag();

        NDArray expected = array(new double[]{
                1, 0, 0,
                0, 2, 0,
                0, 0, 3,

                4, 0, 0,
                0, 5, 0,
                0, 0, 6
        }).reshape(2, 3, 3);

        MatrixTestUtils.assertEqualsMatrix(expected.toDoubles(), diagonal.toDoubles());
    }

    @Test
    public void test3D() {
        NDArray input = array(new double[]{
                1, 2, 3,
                4, 5, 6
        })
                .reshape(2, 1, 3);

        NDArray diagonal = input.diag();

        NDArray expected = array(new double[]{
                1, 0, 0,
                0, 2, 0,
                0, 0, 3,

                4, 0, 0,
                0, 5, 0,
                0, 0, 6
        }).reshape(2, 1, 3, 3);

        MatrixTestUtils.assertEqualsMatrix(expected.toDoubles(), diagonal.toDoubles());
    }

    @Test
    public void test4D() {
        NDArray input = array(new double[]{
                1, 2, 3,
                4, 5, 6,
        })
                .reshape(2, 1, 1, 3);

        NDArray diagonal = input.diag();

        NDArray expected = array(new double[]{
                1, 0, 0,
                0, 2, 0,
                0, 0, 3,
                4, 0, 0,
                0, 5, 0,
                0, 0, 6
        }).reshape(2, 1, 1, 3, 3);

        MatrixTestUtils.assertEqualsMatrix(expected.toDoubles(), diagonal.toDoubles());
    }

}
