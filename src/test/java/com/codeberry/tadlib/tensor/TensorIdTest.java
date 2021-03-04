package com.codeberry.tadlib.tensor;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static java.lang.Math.abs;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class TensorIdTest {

    @Test
    public void idWraps() {
        Tensor.IdGenerator.NEXT_ID.set(Long.MAX_VALUE - 1);

        assertEquals(Long.MAX_VALUE - 1, Tensor.IdGenerator.nextId());
        assertEquals(Long.MAX_VALUE, Tensor.IdGenerator.nextId());
        assertEquals(0, Tensor.IdGenerator.nextId(), "should wrap to zero");
        assertEquals(1, Tensor.IdGenerator.nextId());
    }

    @Test
    public void simple() {
        assertEquals(-1, Tensor.IdGenerator.compareIdNaturalOrder(0, 1));
        assertEquals(1, Tensor.IdGenerator.compareIdNaturalOrder(100, 0));
        assertEquals(0, Tensor.IdGenerator.compareIdNaturalOrder(10, 10));
    }

    @Test
    public void randomSimpleValuesWithinWrappingBounds() {
        Random r = new Random(3);
        long safeBound = Integer.MAX_VALUE / 2;
        long safeAdd = Integer.MAX_VALUE / 4;
        for (int i = 0; i < 20000; i++) {
            long low = abs(r.nextLong()) % safeBound;
            long add = abs(r.nextLong()) % safeAdd;
            long high = low + add;

            assertEquals(-1, Tensor.IdGenerator.compareIdNaturalOrder(low, high), low + " vs " + high);
        }
    }

    @Test
    public void randomOutOfBounds() {
        Random r = new Random(3);
        long safeBound = Integer.MAX_VALUE / 2;
        long safeAdd = 1000;
        for (int i = 0; i < 20000; i++) {
            long low = abs(r.nextLong()) % safeBound;
            long add = abs(r.nextLong()) % safeAdd;
            long highBeyondBounds = low + Integer.MAX_VALUE + 1 + add;

            assertEquals(-1, Tensor.IdGenerator.compareIdNaturalOrder(highBeyondBounds, low), low + " vs " + highBeyondBounds);
        }
    }

    @Test
    public void wrapBounds() {
        long intMax = Integer.MAX_VALUE;
        long longMax = Long.MAX_VALUE;

        assertEquals(-1, Tensor.IdGenerator.compareIdNaturalOrder(0, intMax - 1));
        assertEquals(-1, Tensor.IdGenerator.compareIdNaturalOrder(intMax, intMax * 2));

        assertEquals(-1, Tensor.IdGenerator.compareIdNaturalOrder(longMax - intMax, longMax));
        assertEquals(1, Tensor.IdGenerator.compareIdNaturalOrder(longMax, longMax - intMax));
    }

    @Test
    public void wraps() {
        long intMax = Integer.MAX_VALUE;
        long longMax = Long.MAX_VALUE;

        assertEquals(-1, Tensor.IdGenerator.compareIdNaturalOrder(longMax, 0));
        assertEquals(1, Tensor.IdGenerator.compareIdNaturalOrder(0, longMax));

        assertEquals(-1, Tensor.IdGenerator.compareIdNaturalOrder(longMax, longMax - intMax - 1));
        assertEquals(1, Tensor.IdGenerator.compareIdNaturalOrder(longMax - 1, longMax - intMax - 1));
        assertEquals(-1, Tensor.IdGenerator.compareIdNaturalOrder(longMax - 1, longMax - intMax - 1 - 1));


    }

}
