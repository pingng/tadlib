package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.provider.java.NDArray;
import org.junit.jupiter.api.Test;

import static com.codeberry.tadlib.array.TArrayFactory.range;
import static com.codeberry.tadlib.provider.ProviderStore.array;
import static com.codeberry.tadlib.util.MatrixTestUtils.assertEqualsMatrix;

public class TensorConcatTest {

    @Test
    public void concat2Tensors() {
        Tensor a = new Tensor(range(2 * 3 * 4).reshape(2, 3, 4));
        Tensor b = new Tensor(range(1 * 3 * 4).reshape(1, 3, 4).mul(10));

        Tensor concat = Ops.concat(0, a, b);
        NDArray fakeGradient = range(3 * 3 * 4).reshape(3, 3, 4);
        concat.backward(fakeGradient);

        assertEqualsMatrix(array(new double[]{
                        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
                        14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 0.0, 10.0, 20.0, 30.0,
                        40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0
                }).reshape(3, 3, 4).toDoubles(),
                concat.val().toDoubles());
        assertEqualsMatrix(array(new double[]{
                        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
                        18.0, 19.0, 20.0, 21.0, 22.0, 23.0
                }).reshape(2, 3, 4).toDoubles(),
                a.grad().toDoubles());
        assertEqualsMatrix(array(new double[]{
                        24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0
                }).reshape(1, 3, 4).toDoubles(),
                b.grad().toDoubles());
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
                        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                        12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
                        0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0,
                        0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0
                }).reshape(4, 3, 4).toDoubles(),
                concat.val().toDoubles());
        assertEqualsMatrix(array(new double[]{
                        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                        12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0
                }).reshape(2, 3, 4).toDoubles(),
                a.grad().toDoubles());
        assertEqualsMatrix(array(new double[]{
                        24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0
                }).reshape(1, 3, 4).toDoubles(),
                b.grad().toDoubles());
        assertEqualsMatrix(array(new double[]{
                        36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0
                }).reshape(1, 3, 4).toDoubles(),
                c.grad().toDoubles());
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
                        0.0, 1.0, 2.0, 3.0, 0.0, 10.0, 20.0, 30.0, 0.0, 100.0, 200.0, 300.0,

                        4.0, 5.0, 6.0, 7.0, 40.0, 50.0, 60.0, 70.0, 400.0, 500.0, 600.0, 700.0,
                }).reshape(1, 2, 4 * 3).toDoubles(),
                concat.val().toDoubles());
        assertEqualsMatrix(array(new double[]{
                        0.0, 1.0, 2.0, 3.0, 12.0, 13.0, 14.0, 15.0
                }).reshape(1, 2, 4).toDoubles(),
                a.grad().toDoubles());
        assertEqualsMatrix(array(new double[]{
                        4.0, 5.0, 6.0, 7.0, 16.0, 17.0, 18.0, 19.0
                }).reshape(1, 2, 4).toDoubles(),
                b.grad().toDoubles());
        assertEqualsMatrix(array(new double[]{
                        8.0, 9.0, 10.0, 11.0, 20.0, 21.0, 22.0, 23.0
                }).reshape(1, 2, 4).toDoubles(),
                c.grad().toDoubles());
    }
}
