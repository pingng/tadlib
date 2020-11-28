package com.codeberry.tadlib.singlevalue.example;

import com.codeberry.tadlib.singlevalue.Value;

import java.util.Random;

import static com.codeberry.tadlib.singlevalue.Ops.*;
import static com.codeberry.tadlib.singlevalue.Value.constant;
import static com.codeberry.tadlib.singlevalue.Value.value;

public class SimpleExample {

    public static final double LEARNING_RATE = 0.01;
    public static final int EPOCHS = 3000;

    public static void main(String[] args) {
        Random rand = new Random(6);
        Value a = newParam(rndZeroMean(rand));
        Value b = newParam(rndZeroMean(rand));

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            // sumLoss = ((a + b) - 11)^2
            Value sumLoss = sqr(sub(add(a, b), constant(11)));

            // productLoss = ((a * b) - 18)^2
            Value productLoss = sqr(sub(mul(a, b), constant(18)));

            Value loss = add(sumLoss, productLoss);

            loss.backward();

            // Update params
            a = value(a.v - a.grad * LEARNING_RATE);
            b = value(b.v - b.grad * LEARNING_RATE);

            if ((epoch % 500) == 0) {
                System.out.println(epoch + ": loss: " + loss + " a: " + a + " b: " + b);
            }
        }
        System.out.println("Final:  a: " + a + " b: " + b);
    }

    private static Value newParam(double v) {
        return value(v);
    }

    private static double rndZeroMean(Random rand) {
        return rand.nextDouble() - 0.5;
    }

}