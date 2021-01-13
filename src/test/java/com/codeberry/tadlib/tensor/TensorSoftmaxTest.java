package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.array.NDArray;
import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.opencl.OpenCLProvider;
import com.codeberry.tadlib.util.MatrixTestUtils;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static java.util.Arrays.deepToString;

public class TensorSoftmaxTest {
    @BeforeEach
    public void init() {
//        ProviderStore.setProvider(new JavaProvider()); enableMultiThreading();
        ProviderStore.setProvider(new OpenCLProvider());
    }

    @Test
    public void softmax() {
        Tensor input = new Tensor(ProviderStore.array(new double[]{0.234, 2.73, -5.3, 2, 2.92, 0.2})
                .reshape(1, -1));
        Tensor labels = new Tensor(ProviderStore.array(new double[]{1., 0., 0., 0, 0, 0})
                .reshape(1, -1));

        NDArray softmax = input.getVals().softmax();

        MatrixTestUtils.assertEqualsMatrix(ProviderStore.array(new double[]{
                2.88811444e-02, 3.50439801e-01, 1.14085049e-04, 1.68880090e-01, 4.23769188e-01, 2.79156912e-02
        }).reshape(1, -1).toDoubles(), softmax.toDoubles());

        double backpropGrad = 0.85;
        Tensor cost = Ops.sumSoftmaxCrossEntropy(labels, input);
        cost.backward(ProviderStore.array(backpropGrad));

        Assertions.assertEquals(3.5445664, (double) cost.getVals().toDoubles(), 0.000001);

        NDArray expectedGrad = ProviderStore.array(new double[]{
                -9.71118867e-01, 3.50439727e-01, 1.14085015e-04, 1.68880060e-01, 4.23769146e-01, 2.79156882e-02
        }).reshape(1, -1).mul(backpropGrad);
        System.out.println(deepToString((Object[]) expectedGrad.toDoubles()));
        System.out.println(deepToString((Object[]) input.getGradient().toDoubles()));
        MatrixTestUtils.assertEqualsMatrix(expectedGrad.toDoubles(), input.getGradient().toDoubles());

    }
}
