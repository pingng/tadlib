package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.array.NDArray;
import com.codeberry.tadlib.provider.opencl.OpenCLProvider;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static com.codeberry.tadlib.array.TArrayFactory.*;
import static com.codeberry.tadlib.provider.ProviderStore.array;
import static com.codeberry.tadlib.provider.ProviderStore.setProvider;
import static com.codeberry.tadlib.util.MatrixTestUtils.assertEqualsMatrix;

public class TensorConcatTest {
    @BeforeEach
    public void init() {
//        setProvider(new JavaProvider());
        setProvider(new OpenCLProvider());
    }

    @Test
    public void concat2Tensors() {
        Tensor a = new Tensor(range(2 * 3 * 4).reshape(2, 3, 4));
        Tensor b = new Tensor(range(1 * 3 * 4).reshape(1, 3, 4).mul(10));

        Tensor concat = Ops.concat(0, a, b);
        NDArray fakeGradient = range(3 * 3 * 4).reshape(3, 3, 4);
        concat.backward(fakeGradient);

        assertEqualsMatrix(array(new double[]{
                        0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.,
                        14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 0., 10., 20., 30.,
                        40., 50., 60., 70., 80., 90., 100., 110.
                }).reshape(3, 3, 4).toDoubles(),
                concat.getVals().toDoubles());
        assertEqualsMatrix(array(new double[]{
                        0. , 1. , 2. , 3. , 4. , 5.,  6.,  7. , 8. , 9., 10., 11., 12., 13., 14., 15., 16., 17.,
                        18., 19., 20., 21., 22., 23.
                }).reshape(2, 3, 4).toDoubles(),
                a.getGradient().toDoubles());
        assertEqualsMatrix(array(new double[]{
                        24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35.
                }).reshape(1, 3, 4).toDoubles(),
                b.getGradient().toDoubles());
    }

    @Test
    public void concat3Tensors() {
        Tensor a = new Tensor(range(2 * 3 * 4).reshape(2, 3, 4));
        Tensor b = new Tensor(range(1 * 3 * 4).reshape(1, 3, 4).mul(10));
        Tensor c = new Tensor(range(1 * 3 * 4).reshape(1, 3, 4).mul(100));

        Tensor concat = Ops.concat(0, a, b, c);
        NDArray fakeGradient = range(4 * 3 * 4).reshape(4, 3, 4);
        concat.backward(fakeGradient);

        assertEqualsMatrix(array(new double[]{
                        0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.,
                        12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23.,
                        0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110.,
                        0., 100., 200., 300., 400., 500., 600., 700., 800., 900., 1000., 1100.
                }).reshape(4, 3, 4).toDoubles(),
                concat.getVals().toDoubles());
        assertEqualsMatrix(array(new double[]{
                        0. , 1. , 2. , 3. , 4. , 5.,  6.,  7. , 8. , 9., 10., 11.,
                        12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23.
                }).reshape(2, 3, 4).toDoubles(),
                a.getGradient().toDoubles());
        assertEqualsMatrix(array(new double[]{
                        24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35.
                }).reshape(1, 3, 4).toDoubles(),
                b.getGradient().toDoubles());
        assertEqualsMatrix(array(new double[]{
                        36., 37., 38., 39., 40., 41., 42., 43., 44., 45., 46., 47.
                }).reshape(1, 3, 4).toDoubles(),
                c.getGradient().toDoubles());
    }

    @Test
    public void concat3Tensors_OnAxis1() {
        Tensor a = new Tensor(range(1 * 2 * 4).reshape(1, 2, 4));
        Tensor b = new Tensor(range(1 * 2 * 4).reshape(1, 2, 4).mul(10));
        Tensor c = new Tensor(range(1 * 2 * 4).reshape(1, 2, 4).mul(100));

        Tensor concat = Ops.concat(-1, a, b, c);
        NDArray fakeGradient = range(1 * 2 * (4 * 3)).reshape(1, 2, 4 * 3);
        concat.backward(fakeGradient);

        assertEqualsMatrix(array(new double[]{
                        0., 1., 2., 3., 0., 10., 20., 30., 0., 100., 200., 300.,

                        4., 5., 6., 7., 40., 50., 60., 70., 400., 500., 600., 700.,
                }).reshape(1, 2, 4*3).toDoubles(),
                concat.getVals().toDoubles());
        assertEqualsMatrix(array(new double[]{
                        0. , 1. , 2. , 3. , 12., 13., 14., 15.
                }).reshape(1, 2, 4).toDoubles(),
                a.getGradient().toDoubles());
        assertEqualsMatrix(array(new double[]{
                        4., 5., 6., 7., 16., 17., 18., 19.
                }).reshape(1, 2, 4).toDoubles(),
                b.getGradient().toDoubles());
        assertEqualsMatrix(array(new double[]{
                        8., 9., 10., 11., 20., 21., 22., 23.
                }).reshape(1, 2, 4).toDoubles(),
                c.getGradient().toDoubles());
    }
}
