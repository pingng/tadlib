package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.array.TArray;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Random;

import static com.codeberry.tadlib.array.TArrayFactory.*;
import static com.codeberry.tadlib.tensor.Ops.*;

public class TensorTrainSimple {
    @Test
    public void testMethod() {
        Random rand = new Random(3);
        Tensor x_data = new Tensor(random(rand,100, 3));
        TArray coeff = list(5, -2, 3.5);
        TArray yM = x_data.vals.matmul(coeff).add(5.0);
        Tensor y_data = new Tensor(yM.reshape(100, 1));

        Random wR = new Random(3);
        Tensor w = new Tensor(random(wR, 3, 1));
        Tensor b = new Tensor(random(wR, 1));

        double lr = 0.005;

        for (int i = 0; i < 1000; i++) {
            Tensor matmuled = matmul(x_data, w);
            Tensor y = add(matmuled, b);
            Tensor diff = sub(y, y_data);
            Tensor squared = mul(diff, diff);
            Tensor sum = sum(squared);

            sum.backward(value(1));

            System.out.println(sum.toDoubles());
            System.out.println(Arrays.deepToString((Object[]) w.gradient.toDoubles()));

            w = new Tensor(w.vals.sub(w.gradient.mul(lr)));
            b = new Tensor(b.vals.sub(b.gradient.mul(lr)));
        }

        System.out.println("======");
        System.out.println(Arrays.deepToString((Object[]) w.toDoubles()));
        System.out.println(Arrays.toString((double[]) b.toDoubles()));
    }
    @Test
    public void testMethod2() {
        Random rand = new Random(3);
        Tensor x_data = new Tensor(random(rand,100, 3));
        Tensor coeff = new Tensor(new double[]{4, -2, 7});
        Tensor y_data = new Tensor(x_data.vals.matmul(coeff.vals));

        Random wR = new Random(3);
        TArray mW = new TArray(random(wR, 3)).reshape(3, 1);
        Tensor w = new Tensor(mW);
        Tensor b = new Tensor(random(wR, 1));

        for (int i = 0; i < 1; i++) {
            Tensor matmuled = matmul(x_data, w);

            double[][] grad = new double[100][1];
            for (double[] doubles : grad) {
                Arrays.fill(doubles, 1);
            }
            matmuled.backward(array(grad));

            System.out.println(matmuled.vals.shape);
            //System.out.println(Arrays.toString((double[]) matmuled.toArray()));
            //System.out.println(Arrays.toString((double[]) w.gradient.toArray()));
        }
    }

    private static double[][] random(Random rand, int rows, int len) {
        double[][] vals = new double[rows][];
        for (int j = 0, valsLength = vals.length; j < valsLength; j++) {
            vals[j] = random(rand, len);
        }
        return vals;
    }

    public static double[] random(Random rand, int len) {
        double[] row = new double[len];
        for (int i = 0; i < row.length; i++) {
            row[i] = rand.nextDouble();
        }
        return row;
    }
}
