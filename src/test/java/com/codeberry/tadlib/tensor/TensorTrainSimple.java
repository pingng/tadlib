package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.java.NDArray;
import com.codeberry.tadlib.provider.java.JavaProvider;
//import com.codeberry.tadlib.provider.opencl.OpenCLProvider;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Random;

import static com.codeberry.tadlib.provider.ProviderStore.array;
import static com.codeberry.tadlib.tensor.Ops.*;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TensorTrainSimple {
    @BeforeEach
    public void init() {
//        ProviderStore.setProvider(new JavaProvider()); enableMultiThreading();
        ProviderStore.setProvider(new JavaProvider());
    }

    @Test
    public void testMethod() {
        Random rand = new Random(3);
        Tensor x_data = new Tensor(random(rand,100, 3));
        NDArray yM = x_data.val().matmul(ProviderStore.array(new double[]{5, -2, 3.5})).add(5.0);
        Tensor y_data = new Tensor(yM.reshape(100, 1));

        Random wR = new Random(3);
        Tensor w = new Tensor(random(wR, 3, 1));
        Tensor b = new Tensor(random(wR, 1));

        double lr = 0.0001;

        Tensor y = ADD(MATMUL(x_data, w), b);
        Tensor diff = SUB(y, y_data);
        Tensor diffSq = MUL(diff, diff);
        Tensor diffSqSum = SUM(diffSq);

        double err = Double.POSITIVE_INFINITY;
        for (int i = 0; i < 1000; i++) {

            var Y = diffSqSum.val();

            err = Y.dataAt(0);

            diffSqSum.backward();

            //System.out.println(Y.toDoubles());
            //System.out.println(Arrays.deepToString((Object[]) w.getGradient().toDoubles()));

            NDArray dw = w.val().sub(w.grad().mul(lr));
            w.set(dw);
            NDArray db = b.val().sub(b.grad().mul(lr));
            b.set(db);
        }

//        System.out.println(Arrays.deepToString((Object[]) w.toDoubles()));
//        System.out.println(Arrays.toString((double[]) b.toDoubles()));
        assertTrue(err < 0.1f);
    }

    @Test
    public void testMethod2() {
        Random rand = new Random(3);
        Tensor x_data = new Tensor(random(rand,100, 3));
        Tensor coeff = new Tensor(new double[]{4, -2, 7});
        Tensor y_data = new Tensor(x_data.val().matmul(coeff.val()));

        Random wR = new Random(3);
        NDArray mW = ProviderStore.array(random(wR, 3)).reshape(3, 1);
        Tensor w = new Tensor(mW);
        Tensor b = new Tensor(random(wR, 1));

        for (int i = 0; i < 1; i++) {
            Tensor matmuled = matmul(x_data, w);

            double[][] grad = new double[100][1];
            for (double[] doubles : grad) {
                Arrays.fill(doubles, 1);
            }
            matmuled.backward(ProviderStore.array(grad));

            NDArray ndArray = matmuled.val();
            System.out.println(ndArray.shape);
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
