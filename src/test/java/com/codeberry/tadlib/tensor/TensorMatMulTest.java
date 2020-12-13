package com.codeberry.tadlib.tensor;

import org.junit.jupiter.api.Test;

import static com.codeberry.tadlib.array.TArrayFactory.array;
import static com.codeberry.tadlib.tensor.Tensor.tensor;
import static java.util.Arrays.deepEquals;
import static java.util.Arrays.deepToString;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TensorMatMulTest {
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

        Object cOut = c.vals.toDoubles();
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
        System.out.println(deepToString((Object[]) a.gradient.toDoubles()));
        System.out.println(deepToString((Object[]) b.gradient.toDoubles()));
        System.out.println(deepToString((Object[]) c.gradient.toDoubles()));

        assertTrue(deepEquals(new double[][] {
                {11, 15},
                {11, 15}
        }, (Object[]) a.gradient.toDoubles()));
        assertTrue(deepEquals(new double[][] {
                {4, 4},
                {6, 6}
        }, (Object[]) b.gradient.toDoubles()));
        assertTrue(deepEquals(gradient, (Object[]) c.gradient.toDoubles()));
    }
}