package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.nn.Initializer;
import com.codeberry.tadlib.nn.optimizer.Optimizer;
import com.codeberry.tadlib.nn.optimizer.RMSProp;
import com.codeberry.tadlib.nn.optimizer.SGD;
import com.codeberry.tadlib.nn.optimizer.schedule.FixedLearningRate;
import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.java.NDArray;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Random;

import static com.codeberry.tadlib.tensor.Ops.*;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TensorTrainSimple {

    @Test
    public void raw() {
        Random rng = new Random(3);

        Tensor x_data = new Tensor(random(rng, 100, 3));
        Tensor y_data = new Tensor(x_data.val().matmul(ProviderStore.array(new double[]{5, -2, 3.5})).add(5.0).reshape(100, 1));

        Tensor w = new Tensor(random(rng, 3, 1));
        Tensor b = new Tensor(random(rng, 1));

        double lr = 0.0001;

        Tensor y = ADD(MATMUL(x_data, w), b);
        Tensor diff = SUB(y, y_data);
        Tensor diffSq = MUL(diff, diff);
        Tensor diffSqSum = SUM(diffSq);

        double err = Double.POSITIVE_INFINITY;
        for (int i = 0; i < 1000; i++) {

            err = diffSqSum.val().scalar();

            diffSqSum.backward();

            //System.out.println(err);
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

    @Test void optimizerSGD_1_layer() {
        testOptimizer(new SGD(new FixedLearningRate(0.0001)), false);
    }

    @Test void optimizerSGD_2_layer() {
        testOptimizer(new SGD(new FixedLearningRate(0.00001)), true);
    }

    @Test void optimizerRMSProp_1_layer() {
        testOptimizer(new RMSProp(new FixedLearningRate(0.01f)), false);
    }
    @Test void optimizerRMSProp_2_layer() {
        testOptimizer(new RMSProp(new FixedLearningRate(0.003f)), true);
    }

    private static void testOptimizer(Optimizer opt, boolean twoLayers) {
        Random rng = new Random(3);

        Tensor x_data = new Tensor(random(rng, 100, 3));
        Tensor y_data = new Tensor(x_data.val().matmul(ProviderStore.array(new double[]{5, -2, 3.5})).add(5.0).reshape(100, 1));

        Tensor y = !twoLayers ?
            DENSE(x_data, 1, true)
            :
            RELU( DENSE( RELU( DENSE(x_data, 4)), 1));
            //DENSE(DENSE(x_data, 4, true), 1, true);

        Tensor diff = SUB(y, y_data);
        Tensor diffSq = MUL(diff, diff);
        Tensor diffSqSum = SUM(diffSq);

        diffSqSum.init(new Initializer.UniformInitializer(rng, 0.1f));

        double err = Double.POSITIVE_INFINITY;
        for (int i = 0; i < 1000; i++) {
            err = diffSqSum.optimize(opt).scalar(); //System.out.println(err);
        }

        double ERR = err;
        assertTrue(err < 0.1f, ()->"err: " + ERR);
    }

    @Deprecated @Test
    public void immediate() {
        Random rng = new Random(3);
        Tensor x_data = new Tensor(random(rng, 100, 3));
        Tensor coeff = new Tensor(new double[]{4, -2, 7});
        //Tensor y_data = new Tensor(x_data.val().matmul(coeff.val()));

        NDArray mW = ProviderStore.array(random(rng, 3)).reshape(3, 1);
        Tensor w = new Tensor(mW);
        Tensor b = new Tensor(random(rng, 1));

        for (int i = 0; i < 1; i++) {
            Tensor matmuled = matmul(x_data, w);

            double[][] grad = new double[100][1];
            for (double[] doubles : grad) {
                Arrays.fill(doubles, 1);
            }
            matmuled.backward(ProviderStore.array(grad));

            //NDArray ndArray = matmuled.val();
            //System.out.println(ndArray.shape);
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
