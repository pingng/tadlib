package com.codeberry.tadlib.singlevalue;

import java.util.Arrays;
import java.util.Random;

import static com.codeberry.tadlib.singlevalue.Ops.*;
import static com.codeberry.tadlib.singlevalue.Value.constant;
import static com.codeberry.tadlib.singlevalue.Value.value;

public class MinimizeExample {

    public static final double LEARNING_RATE = 0.02;
    public static final int EPOCHS = 200;

    public static void main(String[] args) {
        Value target = constant(5.3);
        Value[] params = createParams(4);

        String initialParamsAsStr = Arrays.toString(params);

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            Value y = predict(params);

            Value loss = calcLoss(target, y);
            loss.backward();

            updateParams(LEARNING_RATE, params);

            if ((epoch % 10) == 0) {
                System.out.println(epoch + ": loss: " + loss + " y: " + y);
            }
        }

        System.out.println("===");
        System.out.println("Params(rnd init): " + initialParamsAsStr);
        System.out.println("Params(optmized): " + Arrays.toString(params));
        System.out.println("Predict: " + predict(params));
        System.out.println("Target: " + target);
    }

    private static Value[] createParams(int count) {
        Random rand = new Random(3);
        Value[] params = new Value[count];
        Arrays.setAll(params, _i -> value(rndZeroMean(rand)));
        return params;
    }

    private static double rndZeroMean(Random rand) {
        return rand.nextDouble() - 0.5;
    }

    private static void updateParams(double learningRate, Value... params) {
        for (int j = 0; j < params.length; j++) {
            Value param = params[j];
            params[j] = value(param.v - param.grad * learningRate);
        }
    }

    private static Value calcLoss(Value targeValue, Value y) {
        Value diff = sub(targeValue, y);
        return sqr(diff);
    }

    private static Value predict(Value[] params) {
        Value a = mul(pow(params[0], params[3]), params[1]);
        Value b = add(mul(sin(a), value(3)), params[0]);
        return mul(tanh(params[2]), mul(b, params[1]));
    }
}