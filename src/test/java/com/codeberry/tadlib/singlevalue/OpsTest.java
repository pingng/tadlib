package com.codeberry.tadlib.singlevalue;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;

import static com.codeberry.tadlib.singlevalue.Ops.*;
import static com.codeberry.tadlib.singlevalue.Value.value;
import static org.junit.jupiter.api.Assertions.*;

class OpsTest {
    @Test
    public void testAdd() {
        Value a = value(2.0);
        Value b = value(5.0);

        Value y = add(a, b);

        assertEquals(7.0, y.v);
        y.backward(3);

        assertEquals(3., a.grad);
        assertEquals(3., b.grad);
    }

    @Test
    public void testMul() {
        Value a = value(2.0);
        Value b = value(5.0);

        Value y = mul(a, b);

        assertEquals(10.0, y.v);
        y.backward(4);

        assertEquals(20.0, a.grad);
        assertEquals(8.0, b.grad);
    }

    @Test
    public void testMinimize() {
        Random rand = new Random(4);
        Value[] params = new Value[]{
                value(rand.nextDouble()),
                value(rand.nextDouble()),
                value(rand.nextDouble()),
                value(rand.nextDouble())
        };

        for (int i = 0; i < 100; i++) {
            Value a = mul(pow(params[0], params[3]), params[1]);
            Value b = add(mul(sin(a), value(3)), params[0]);
            Value y = mul(tanh(params[2]), mul(b, params[1]));

            Value diff = sub(value(5.0), y);
            Value loss = sqr(diff);
            System.out.println("loss: " + loss.v + " y: " + y.v);
            System.out.println("\t"+Arrays.toString(params));

            loss.backward(1);

            for (int j = 0; j < params.length; j++) {
                Value param = params[j];
                params[j] = value(param.v - param.grad * 0.01);
            }
        }

    }

    @Test
    public void testNumericDiff() {
        NumDiff numDiff = new NumDiff(4, params -> {
            Value a = mul(pow(params[0], params[3]), params[1]);
            Value b = add(mul(sin(a), value(3)), params[0]);
            Value y = mul(tanh(params[2]), mul(sqr(b), params[1]));

            double _a = Math.pow(params[0].v, params[3].v) * params[1].v;
            double _b = Math.sin(_a) * 3 + params[0].v;
            double _y = Math.tanh(params[2].v) * (_b * _b * params[1].v);

            System.out.println("_y = " + _y + " - "+y.v);
            assertEquals(_y, y.v, 0.00001);

            return y;
        });

        numDiff.test(new Random(3),
                new int[]{21, 23, 25});
    }

    static class NumDiff {
        private final int paramCount;
        private final double[] fixedParams;
        private Function<Value[], Value> graphFn;

        NumDiff(int paramCount, Function<Value[], Value> graphFn) {
            this.paramCount = paramCount;
            this.graphFn = graphFn;
            this.fixedParams = null;
        }

        NumDiff(double[] fixedParams, Function<Value[], Value> graphFn) {
            this.paramCount = fixedParams.length;
            this.fixedParams = fixedParams;
            this.graphFn = graphFn;
        }

        void test(Random rand, int[] ignoreIndex) {
            Arrays.sort(ignoreIndex);

            for (int i = 0; i < 50; i++) {
                System.out.println("=== " + i);
                double[] paramVals = fixedParams == null ? createParams(rand) : fixedParams;
                if (Arrays.binarySearch(ignoreIndex, i)>=0) {
                    System.out.println("IGNORED");
                    continue;
                }

                for (int j = 0; j < paramVals.length; j++) {
                    double numericDiff = calcNumericDiff(paramVals, j);
                    double autograd = getAutogradDiff(paramVals, j);

                    System.out.println(j + ": " + numericDiff + "\n   " + autograd);
                    assertEquals(numericDiff, autograd, 0.000001,
                            Arrays.toString(paramVals));
                }
            }
        }

        private double getAutogradDiff(double[] paramVals, int j) {
            Value[] params = toValues(paramVals);
            Value y = graphFn.apply(params);
            y.backward(1.0);
            System.out.println("y=" + y.v);

            return params[j].grad;
        }

        /**
         * "Neural Networks Demystified [Part 5: Numerical Gradient Checking]"
         * https://youtu.be/pHMzNW8Agq4
         */
        private double calcNumericDiff(double[] paramVals, int j) {
            double e = 0.0001;
            Value[] params = toValues(paramVals);
            params[j] = value(params[j].v - e);
            Value before = graphFn.apply(params);

            Value[] paramsCopy = toValues(paramVals);
            paramsCopy[j] = value(paramsCopy[j].v + e);
            Value after = graphFn.apply(paramsCopy);

            return (after.v - before.v) / (2*e);
        }

        private Value[] toValues(double[] vals) {
            Value[] cp = new Value[vals.length];
            for (int i = 0; i < vals.length; i++) {
                cp[i] = value(vals[i]);
            }
            return cp;
        }

        private double[] createParams(Random rand) {
            double[] params = new double[paramCount];
            for (int j = 0; j < params.length; j++) {
                params[j] = rand.nextDouble();
            }
            return params;
        }
    }
}