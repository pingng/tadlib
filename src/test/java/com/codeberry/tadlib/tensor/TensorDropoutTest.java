package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.opencl.OpenCLProvider;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;

import static com.codeberry.tadlib.array.TArrayFactory.ones;
import static com.codeberry.tadlib.array.TArrayFactory.randomWeight;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class TensorDropoutTest {
    @BeforeEach
    public void init() {
//        ProviderStore.setProvider(new JavaProvider()); enableMultiThreading();
        ProviderStore.setProvider(new OpenCLProvider());
    }

    @Test
    public void dropout() {
        Random rand = new Random(4);

        Tensor input = new Tensor(randomWeight(rand, 30 * 4 * 4 * 10)
                .reshape(30, 4, 4, 10));

        Random dropRnd = new Random(5);
        Tensor dropout = Ops.dropout(input, dropRnd, 0.35, Ops.RunMode.TRAINING);
        dropout.backward(ones(input.getVals().getShape()));

        AtomicInteger zeroCount = new AtomicInteger();
        forEachZeroElement(dropout, (_idx, isZero) -> {
            if (isZero) {
                zeroCount.incrementAndGet();
            }
        });
        assertEquals(1.0 - 0.35, (double) zeroCount.get() / input.getVals().getShape().getSize(), 0.01);

        double[] gradData = input.getGradient().getInternalData();
        forEachZeroElement(dropout, (idx, isZero) -> {
                assertEquals(isZero, gradData[idx] == 0.0);
        });
        assertEquals(1.0 - 0.35, (double) zeroCount.get() / input.getVals().getShape().getSize(), 0.01);
    }

    private void forEachZeroElement(Tensor dropout, BiConsumer<Integer, Boolean> indexZeroConsumer) {
        double[] internalData = dropout.getVals().getInternalData();
        for (int i = 0, internalDataLength = internalData.length; i < internalDataLength; i++) {
            double dat = internalData[i];
            indexZeroConsumer.accept(i, dat == 0.0);
        }
    }
}
