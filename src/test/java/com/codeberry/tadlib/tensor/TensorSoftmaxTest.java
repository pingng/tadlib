package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.java.NDArray;
import com.codeberry.tadlib.provider.java.JavaProvider;
//import com.codeberry.tadlib.provider.opencl.OpenCLProvider;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static com.codeberry.tadlib.provider.ProviderStore.array;
import static com.codeberry.tadlib.util.MatrixTestUtils.*;
import static java.util.Arrays.deepToString;

public class TensorSoftmaxTest {
    @BeforeEach
    public void init() {
//        ProviderStore.setProvider(new JavaProvider());
        ProviderStore.setProvider(new JavaProvider());
    }

    @Test
    public void softmax1D() {
        Tensor input = new Tensor(array(new double[]{5, 2.4, 3, 4}));

        Tensor softmax = Ops.softmax(input);
        softmax.backward(array(new double[]{123.5, 0.123, 0.5, 2.}));

        assertEqualsMatrix(new double[]{0.63391912, 0.04708344, 0.08579162, 0.23320581},
                softmax.toDoubles());

        assertEqualsMatrix(new double[]{28.33357917, -3.70457745, -6.71783678, -17.91116494},
                input.grad().toDoubles());
    }

    @Test
    public void softmax2D() {
        Tensor input = new Tensor(array(new double[]{
                5, 2.4, 3, 4,
                1.5, -0.4, 7.5, -10
        }).reshape(2, 4));

        Tensor softmax = Ops.softmax(input);
        softmax.backward(array(new double[]{
                123.5, 0.123, 0.5, 2.,
                11.5, -10.75, 11.3, 5.
        }).reshape(2, 4));

        assertEqualsMatrix(array(new double[]{
                        0.63391912, 0.04708344, 0.08579162, 0.23320581,
                        2.47170899e-03, 3.69690101e-04, 9.97158576e-01, 2.50386434e-08
                }).reshape(2, 4).toDoubles(),
                softmax.toDoubles());

        assertEqualsMatrix(array(new double[]{
                        28.33357917, -3.70457745, -6.71783678, -17.91116494,
                        5.13268867e-04, -8.14883583e-03, 7.63572451e-03, -1.57551721e-07
                }).reshape(2, 4).toDoubles(),
                input.grad().toDoubles());
    }

    @Test
    public void softmax3D() {
        Tensor input = new Tensor(array(new double[]{
                5, 2.4, 3, 4,
                1.5, -0.4, 7.5, -10
        }).reshape(2, 1, 4));

        Tensor softmax = Ops.softmax(input);
        softmax.backward(array(new double[]{
                123.5, 0.123, 0.5, 2.,
                11.5, -10.75, 11.3, 5.
        }).reshape(2, 1, 4));

        assertEqualsMatrix(array(new double[]{
                        0.63391912, 0.04708344, 0.08579162, 0.23320581,
                        2.47170899e-03, 3.69690101e-04, 9.97158576e-01, 2.50386434e-08
                }).reshape(2, 1, 4).toDoubles(),
                softmax.toDoubles());

        assertEqualsMatrix(array(new double[]{
                        28.33357917, -3.70457745, -6.71783678, -17.91116494,
                        5.13268867e-04, -8.14883583e-03, 7.63572451e-03, -1.57551721e-07
                }).reshape(2, 1, 4).toDoubles(),
                input.grad().toDoubles());
    }

    @Test
    public void softmax4D() {
        Tensor input = new Tensor(array(new double[]{
                0.5507979, 0.70814782, 0.29090474,
                0.51082761, 0.89294695, 0.89629309,

                0.12558531, 0.20724288, 0.0514672,
                0.44080984, 0.02987621, 0.45683322,

                0.64914405, 0.27848728, 0.6762549,
                0.59086282, 0.02398188, 0.55885409,

                0.25925245, 0.4151012, 0.28352508,
                0.69313792, 0.44045372, 0.15686774
        }).reshape(2, 2, 2, 3));

        Tensor softmax = Ops.softmax(input);
        softmax.backward(array(new double[]{
                0.54464902, 0.78031476, 0.30636353,
                0.22195788, 0.38797126, 0.93638365,
                0.97599542, 0.67238368, 0.90283411,
                0.84575087, 0.37799404, 0.09221701,
                0.6534109, 0.55784076, 0.36156476,
                0.2250545, 0.40651992, 0.46894025,
                0.26923558, 0.29179277, 0.4576864,
                0.86053391, 0.5862529, 0.28348786
        }).reshape(2, 2, 2, 3));

        assertEqualsMatrix(array(new double[]{
                        0.33995809, 0.39788868, 0.26215323,
                        0.25408534, 0.37233335, 0.37358131,

                        0.33182396, 0.36005692, 0.30811912,
                        0.37324806, 0.24747501, 0.37927693,

                        0.36794973, 0.25398865, 0.37806162,
                        0.39435439, 0.22371413, 0.38193148,

                        0.31316361, 0.36597847, 0.32085792,
                        0.42343474, 0.32888732, 0.24767793
                }).reshape(2, 2, 2, 3).toDoubles(),
                softmax.toDoubles());

        assertEqualsMatrix(array(new double[]{
                        -0.01064117, 0.08131425, -0.07067308,
                        -0.08351999, -0.06057679, 0.14409679,

                        0.04375428, -0.06184044, 0.01808616,
                        0.14988005, -0.01638302, -0.13349703,

                        0.04952955, 0.00991557, -0.05944512,
                        -0.05274254, 0.01067595, 0.04206659,

                        -0.02152104, -0.0168951, 0.03841614,
                        0.09871497, -0.01353433, -0.08518063
                }).reshape(2, 2, 2, 3).toDoubles(),
                input.grad().toDoubles());
    }

    @Test
    public void softmaxCrossEntropy() {
        Tensor input = new Tensor(array(new double[]{0.234, 2.73, -5.3, 2, 2.92, 0.2})
                .reshape(1, -1));
        Tensor labels = new Tensor(array(new double[]{1., 0., 0., 0, 0, 0})
                .reshape(1, -1));

        NDArray softmax = input.val().softmax();

        assertEqualsMatrix(array(new double[]{
                2.88811444e-02, 3.50439801e-01, 1.14085049e-04, 1.68880090e-01, 4.23769188e-01, 2.79156912e-02
        }).reshape(1, -1).toDoubles(), softmax.toDoubles());

        double backpropGrad = 0.85;
        Tensor cost = Ops.sumSoftmaxCrossEntropy(labels, input);
        cost.backward(array(backpropGrad));

        Assertions.assertEquals(3.5445664, (double) cost.val().toDoubles(), 0.000001);

        NDArray expectedGrad = array(new double[]{
                -9.71118867e-01, 3.50439727e-01, 1.14085015e-04, 1.68880060e-01, 4.23769146e-01, 2.79156882e-02
        }).reshape(1, -1).mul(backpropGrad);
        System.out.println(deepToString((Object[]) expectedGrad.toDoubles()));
        System.out.println(deepToString((Object[]) input.grad().toDoubles()));
        assertEqualsMatrix(expectedGrad.toDoubles(), input.grad().toDoubles());

    }
}
