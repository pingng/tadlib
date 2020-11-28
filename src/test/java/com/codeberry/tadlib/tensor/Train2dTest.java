package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.array.TArray;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectWriter;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static com.codeberry.tadlib.tensor.Ops.*;
import static com.codeberry.tadlib.tensor.TensorTrainSimple.random;
import static java.util.Arrays.deepToString;

public class Train2dTest {
    @Test
    public void testMethod() {
        Random rand = new Random(4);
        int examples = 40;
        int out = 1;
        Tensor x_train = new Tensor(randMatrix(rand, examples * 4 * 4 * 1)
                .reshape(examples, 4, 4, 1), Tensor.GradientMode.NONE);
        Tensor y_train = new Tensor(randMatrix(rand, examples * out)
                .reshape(examples, out), Tensor.GradientMode.NONE);

        Random nr = new Random(4);
        int hiddenOut = 3;
        Tensor w = new Tensor(randMatrix(nr, 3 * 3 * 1 * hiddenOut)
                .reshape(3, 3, 1, hiddenOut));
        Tensor b = new Tensor(randMatrix(nr, hiddenOut));


        Tensor fullW = new Tensor(randMatrix(nr, 2 * 2 * hiddenOut * out)
                .reshape(2 * 2 * hiddenOut, out));
        Tensor fullB = new Tensor(randMatrix(nr, out));

        double lrBase = 0.03 / examples;
        for (int i = 0; i <= 5000; i++) {
            double lr = (i < 1000 ? lrBase * 0.2 : lrBase);

            Tensor h_w = conv2d(x_train, w);
            Tensor h = add(h_w, b);
            Tensor maxed = max2d(h, 2);
            Tensor relu = relu(maxed);
            Tensor flattened = flatten(relu);

            Tensor y_w = matmul(flattened, fullW);
            Tensor y = add(y_w, fullB);

            Tensor diff = sub(y, y_train);
            Tensor squared = mul(diff, diff);
            Tensor totalErr = sum(squared);
            totalErr.backward(new Tensor(1).vals);

            if ((i % 500) == 0) {
                System.out.println("=== " + i + " Err: " + totalErr.toDoubles());
                System.out.println("h: " + toJson(h.toDoubles()));
                System.out.println("Flatten: " + toJson(flattened.toDoubles()));
                System.out.println("fW: " + deepToString((Object[]) fullW.toDoubles()));
                System.out.println("  Y_w: " + deepToString((Object[]) y_w.toDoubles()));
                System.out.println("Y: " + deepToString((Object[]) y.toDoubles()));
                System.out.println("y_train: " + deepToString((Object[]) y_train.toDoubles()));
            }

            w = updateParam(w, lr);
            b = updateParam(b, lr);
            fullW = updateParam(fullW, lr);
            fullB = updateParam(fullB, lr);
        }

        //System.out.println(y_b.m.shape);
        //System.out.println(Arrays.deepToString((Object[]) y_b.m.toArray()));
    }

    static ObjectMapper objectMapper = new ObjectMapper();

    private static String toJson(Object toArray) {
        ObjectWriter
                w = objectMapper.writerWithDefaultPrettyPrinter();
        try {
            return w.writeValueAsString(toArray);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    private static Tensor updateParam(Tensor w, double lr) {
        return new Tensor(w.vals.sub(w.gradient.mul(lr)));
        //return new Tensor(w.m.sub(w.gradient.m.mul(lr)));
    }

    private TArray randMatrix(Random rand, int size) {
        return new TArray(random(rand, size));
    }
}
