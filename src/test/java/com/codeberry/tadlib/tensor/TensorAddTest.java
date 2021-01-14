package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.opencl.OpenCLProvider;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static com.codeberry.tadlib.tensor.Tensor.tensor;
import static java.util.Arrays.deepEquals;
import static java.util.Arrays.deepToString;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TensorAddTest {
    @BeforeEach
    public void init() {
//        ProviderStore.setProvider(new JavaProvider());
        ProviderStore.setProvider(new OpenCLProvider());
    }

    @Test
    public void add() {
        Tensor a = tensor(new double[][]
                {
                        {1, 2, 3},
                        {0.1, 0.2, 0.3}
                });
        Tensor b = tensor(new double[][]
                {
                        {5, 7, 11},
                        {13, 17, 19}
                });
        Tensor c = Ops.add(a, b);

        double[][] gradient = {
                {3, 2, 1},
                {6, 5, 4}
        };
        c.backward(ProviderStore.array(gradient));

        assertTrue(deepEquals(gradient, (Object[]) c.getGradient().toDoubles()));
        assertTrue(deepEquals(gradient, (Object[]) a.getGradient().toDoubles()));
        assertTrue(deepEquals(gradient, (Object[]) b.getGradient().toDoubles()));
    }

    @Test
    public void addBroadcast_MissingDims_Right() {
        Tensor a = tensor(new double[][]
                {
                        {1, 2, 3},
                        {0.1, 0.2, 0.3}
                });
        Tensor b = tensor(new double[]{13, 17, 19});
        Tensor c = Ops.add(a, b);

        assertTrue(deepEquals(new double[][]
                {
                        {13+1, 17+2, 19+3},
                        {13+0.1, 17+0.2, 19+0.3}
                }, (Object[]) c.getVals().toDoubles()));

        double[][] gradient = {
                {1, 2, 3},
                {10, 20, 30}
        };
        c.backward(ProviderStore.array(gradient));

        assertTrue(deepEquals(gradient, (Object[]) c.getGradient().toDoubles()));
        assertTrue(deepEquals(gradient, (Object[]) a.getGradient().toDoubles()));
        double[] doubles = {11, 22, 33};
        assertArrayEquals( doubles, (double[]) b.getGradient().toDoubles());
    }

    @Test
    public void addBroadcast_MissingDims_Left() {
        Tensor a = tensor(new double[]{1, 2, 3});
        Tensor b = tensor(new double[][] {
                {5, 7, 11},
                {13, 17, 19}
        });
        Tensor c = Ops.add(a, b);

        assertTrue(deepEquals(new double[][]
                {
                        {5+1, 7+2, 11+3},
                        {13+1, 17+2, 19+3}
                }, (Object[]) c.getVals().toDoubles()));

        double[][] gradient = {
                {1, 2, 3},
                {10, 20, 30}
        };
        c.backward(ProviderStore.array(gradient));

        assertTrue(deepEquals(gradient, (Object[]) c.getGradient().toDoubles()));
        assertTrue(deepEquals(gradient, (Object[]) b.getGradient().toDoubles()));
        double[] doubles = {11, 22, 33};
        assertArrayEquals( doubles, (double[]) a.getGradient().toDoubles());
    }

    @Test
    public void addBroadcast_SingleDimensions() {
        Tensor a = tensor(new double[][]
                {
                        {1, 2, 3},
                        {0.1, 0.2, 0.3}
                });
        Tensor b = tensor(new double[][]{{13, 17, 19}});
        Tensor c = Ops.add(a, b);

        assertTrue(deepEquals(new double[][]
                {
                        {13+1, 17+2, 19+3},
                        {13+0.1, 17+0.2, 19+0.3}
                }, (Object[]) c.getVals().toDoubles()));

        double[][] gradient = {
                {1, 2, 3},
                {10, 20, 30}
        };
        c.backward(ProviderStore.array(gradient));

        assertTrue(deepEquals(gradient, (Object[]) c.getGradient().toDoubles()));
        assertTrue(deepEquals(gradient, (Object[]) a.getGradient().toDoubles()));
        Object[] a2 = (Object[]) b.getGradient().toDoubles();
        System.out.println(deepToString(a2));
        assertTrue(deepEquals(new double[][]{{11, 22, 33}}, a2));
    }
}