package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.java.JavaProvider;
//import com.codeberry.tadlib.provider.opencl.OpenCLProvider;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static com.codeberry.tadlib.provider.ProviderStore.array;
import static com.codeberry.tadlib.tensor.Tensor.tensor;
import static com.codeberry.tadlib.util.MatrixTestUtils.assertEqualsMatrix;
import static java.util.Arrays.deepEquals;
import static java.util.Arrays.deepToString;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TensorMatMulTest {
    @BeforeEach
    public void init() {
//        ProviderStore.setProvider(new JavaProvider());
        ProviderStore.setProvider(new JavaProvider());
    }

    @Test
    public void matMulLeftBroadcast() {
        Tensor a = tensor(array(new double[]{
                1.5, 2.5,
                3.5, 4.5
        }).reshape(1, 2, 2));
        Tensor b = tensor(array(new double[]{
                10, 20,
                30, 40,

                50, 60,
                70, 80
        }).reshape(2, 2, 2));
        Tensor c = Ops.matmul(a, b);

        Object cOut = c.val().toDoubles();
        System.out.println("out shape: " + c.shape());
        System.out.println(deepToString((Object[]) cOut));
        assertEqualsMatrix(
                new double[][][]{
                        {
                                {1.5 * 10 + 2.5 * 30, 1.5 * 20 + 2.5 * 40},
                                {3.5 * 10 + 4.5 * 30, 3.5 * 20 + 4.5 * 40}
                        },
                        {
                                {1.5 * 50 + 2.5 * 70, 1.5 * 60 + 2.5 * 80},
                                {3.5 * 50 + 4.5 * 70, 3.5 * 60 + 4.5 * 80}
                        },
                }, cOut);

        double[] gradient = {
                1, 2,
                3, 5,

                7, 11,
                13, 17
        };
        c.backward(array(gradient).reshape(2, 2, 2));

        System.out.println("---");
        System.out.println(deepToString((Object[]) a.grad().toDoubles()));
        System.out.println(deepToString((Object[]) b.grad().toDoubles()));
        System.out.println(deepToString((Object[]) c.grad().toDoubles()));

        assertEqualsMatrix(new double[][][]{
                {
                        {1060, 1480},
                        {1800, 2560}
                }
        }, a.grad().toDoubles());
        assertEqualsMatrix(new double[][][]{
                {
                        {12, 20.5},
                        {16, 27.5}
                },
                {
                        {56, 76},
                        {76, 104}
                }
        }, b.grad().toDoubles());
    }

    @Test
    public void matMulRightBroadcast() {
        Tensor a = tensor(array(new double[]{
                11, 23,
                35, 47,

                61, 73,
                87, 103
        }).reshape(2, 2, 2));
        Tensor b = tensor(array(new double[]{
                1.5, 2.5,
                3.5, 4.5
        }).reshape(1, 2, 2));
        Tensor c = Ops.matmul(a, b);

        Object cOut = c.val().toDoubles();
        System.out.println("out shape: " + c.shape());
        System.out.println(deepToString((Object[]) cOut));
        assertEqualsMatrix(
                new double[][][]{
                        {
                                {1.5 * 11 + 3.5 * 23, 2.5 * 11 + 4.5 * 23},
                                {1.5 * 35 + 3.5 * 47, 2.5 * 35 + 4.5 * 47}
                        },
                        {
                                {1.5 * 61 + 3.5 * 73, 2.5 * 61 + 4.5 * 73},
                                {1.5 * 87 + 3.5 * 103, 2.5 * 87 + 4.5 * 103}
                        },
                }, cOut);

        double[] gradient = {
                1, 2,
                3, 5,

                7, 11,
                13, 17
        };
        c.backward(array(gradient).reshape(2, 2, 2));

        System.out.println("---");
        System.out.println(deepToString((Object[]) a.grad().toDoubles()));
        System.out.println(deepToString((Object[]) b.grad().toDoubles()));
        System.out.println(deepToString((Object[]) c.grad().toDoubles()));

        assertEqualsMatrix(new double[][][]{
                {
                        {6.5, 12.5},
                        {17, 33}
                },
                {
                        {38, 74},
                        {62, 122}
                }
        }, a.grad().toDoubles());
        assertEqualsMatrix(new double[][][]{
                {
                        {1674, 2347},
                        {2014, 2835}
                }
        }, b.grad().toDoubles());
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

        Object cOut = c.val().toDoubles();
        System.out.println(deepToString((Object[]) cOut));
        assertTrue(deepEquals(new double[][]{
                {19, 22},
                {43, 50}
        }, (Object[]) cOut));

        double[][] gradient = {
                {1, 1},
                {1, 1}
        };
        c.backward(array(gradient));

        System.out.println("---");
        System.out.println(deepToString((Object[]) a.grad().toDoubles()));
        System.out.println(deepToString((Object[]) b.grad().toDoubles()));
        System.out.println(deepToString((Object[]) c.grad().toDoubles()));

        assertTrue(deepEquals(new double[][]{
                {11, 15},
                {11, 15}
        }, (Object[]) a.grad().toDoubles()));
        assertTrue(deepEquals(new double[][]{
                {4, 4},
                {6, 6}
        }, (Object[]) b.grad().toDoubles()));
        assertTrue(deepEquals(gradient, (Object[]) c.grad().toDoubles()));
    }
}