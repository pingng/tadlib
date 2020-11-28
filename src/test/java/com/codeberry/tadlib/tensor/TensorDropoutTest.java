package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.array.TArray;
import org.junit.jupiter.api.Test;

import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TensorDropoutTest {
    @Test
    public void dropout() {
        Random rand = new Random(4);

        Tensor input = new Tensor(TArray.randWeight(rand, 30 * 4 * 4 * 10)
                .reshape(30, 4, 4, 10));

        Random dropRnd = new Random(5);
        Tensor dropout = Ops.dropout(input, dropRnd, 0.35, Ops.RunMode.TRAINING);
        dropout.backward(TArray.ones(input.vals.shape));

        AtomicInteger zeroCount = new AtomicInteger();
        forEachZeroElement(dropout, (_idx, isZero) -> {
            if (isZero) {
                zeroCount.incrementAndGet();
            }
        });
        assertEquals(1.0 - 0.35, (double) zeroCount.get() / input.vals.shape.size, 0.01);

        double[] gradData = input.gradient.getInternalData();
        forEachZeroElement(dropout, (idx, isZero) -> {
                assertEquals(isZero, gradData[idx] == 0.0);
        });
        assertEquals(1.0 - 0.35, (double) zeroCount.get() / input.vals.shape.size, 0.01);
    }

    private void forEachZeroElement(Tensor dropout, BiConsumer<Integer, Boolean> indexZeroConsumer) {
        double[] internalData = dropout.vals.getInternalData();
        for (int i = 0, internalDataLength = internalData.length; i < internalDataLength; i++) {
            double dat = internalData[i];
            indexZeroConsumer.accept(i, dat == 0.0);
        }
    }
}
