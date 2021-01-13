package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.opencl.OpenCLProvider;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static com.codeberry.tadlib.tensor.Tensor.tensor;
import static java.util.Arrays.deepEquals;
import static java.util.Arrays.deepToString;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TensorMatMulTest {
    @BeforeEach
    public void init() {
//        ProviderStore.setProvider(new JavaProvider()); enableMultiThreading();
        ProviderStore.setProvider(new OpenCLProvider());
    }

    @Test
    public void matmul() {
        Tensor a = tensor(new double[][]
                {
                        {1, 2},
                        {3, 4}
                });
        Tensor b = tensor(new double[][]
                {
                        {5, 6},
                        {7, 8}
                });
        Tensor c = Ops.matmul(a, b);

        Object cOut = c.getVals().toDoubles();
        System.out.println(deepToString((Object[]) cOut));
        assertTrue(deepEquals(new double[][]{
                {19, 22},
                {43, 50}
        }, (Object[]) cOut));

        double[][] gradient = {
                {1, 1},
                {1, 1}
        };
        c.backward(ProviderStore.array(gradient));

        System.out.println("---");
        System.out.println(deepToString((Object[]) a.getGradient().toDoubles()));
        System.out.println(deepToString((Object[]) b.getGradient().toDoubles()));
        System.out.println(deepToString((Object[]) c.getGradient().toDoubles()));

        assertTrue(deepEquals(new double[][] {
                {11, 15},
                {11, 15}
        }, (Object[]) a.getGradient().toDoubles()));
        assertTrue(deepEquals(new double[][] {
                {4, 4},
                {6, 6}
        }, (Object[]) b.getGradient().toDoubles()));
        assertTrue(deepEquals(gradient, (Object[]) c.getGradient().toDoubles()));
    }
}