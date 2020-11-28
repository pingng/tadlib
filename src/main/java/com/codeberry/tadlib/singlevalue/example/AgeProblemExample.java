package com.codeberry.tadlib.singlevalue.example;

import com.codeberry.tadlib.singlevalue.Value;

import java.util.Random;

import static com.codeberry.tadlib.singlevalue.Ops.*;
import static com.codeberry.tadlib.singlevalue.Value.constant;
import static com.codeberry.tadlib.singlevalue.Value.value;

public class AgeProblemExample {

    public static final double LEARNING_RATE = 0.08;
    public static final int EPOCHS = 500;

    public static void main(String[] args) {
        Random rand = new Random(4);
        Value adam = newParam(rndZeroMean(rand));
        Value belle = newParam(rndZeroMean(rand));

        String assignment = "Adam is 24 years older than Belle, but in six years,\n" +
                "Adam will by three times older than Belle.\n" +
                "Q: How old is Adam?";

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            // First criterion:
            //      "Adam is 24 years older than Belle,..."
            //      Adam - 24 = Belle
            Value firstLeft = sub(adam, constant(24));
            Value firstRight = belle;
            Value firstCriteria = sub(firstLeft, firstRight);
            // This should be minimized to zero
            Value firstLoss = sqr(firstCriteria);

            // Second criterion:
            //      "...but in six years, Adam will by three times older than Belle."
            //      (Adam + 6) = (Belle + 6) * 3.
            Value secondLeft = add(adam, constant(6));
            Value secondRight = mul(add(belle, constant(6)), constant(3));
            Value secondCriteria = sub(secondLeft, secondRight);
            // This should also be minimized to zero
            Value secondLoss = sqr(secondCriteria);

            Value loss = add(firstLoss, secondLoss);

            loss.backward();

            // Update params
            adam = value(adam.v - adam.grad * LEARNING_RATE);
            belle = value(belle.v - belle.grad * LEARNING_RATE);

            if ((epoch % 100) == 0) {
                System.out.println(epoch + ": loss: " + loss + " adam: " + adam + " belle: " + belle);
            }
        }

        System.out.println();
        System.out.println('"' + assignment + '"');
        System.out.printf("A: Adam is %.1f years old (while Belle is %.1f).%n", adam.v, belle.v);
    }

    private static Value newParam(double v) {
        return value(v);
    }

    private static double rndZeroMean(Random rand) {
        return rand.nextDouble() - 0.5;
    }

}