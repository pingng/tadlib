package com.codeberry.tadlib.array;

import com.codeberry.tadlib.provider.JavaProvider;
import com.codeberry.tadlib.provider.ProviderStore;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static com.codeberry.tadlib.array.TArrayFactory.*;
import static org.junit.jupiter.api.Assertions.*;

class TArrayAdd {

    @BeforeEach
    public void init() {
        ProviderStore.setProvider(new JavaProvider());
    }

    @Test
    public void plainValue() {
        JavaArray a = new JavaArray(2.0);
        JavaArray b = new JavaArray(6.0);

        JavaArray c = a.add(b);

        assertEquals(8.0, (double) c.toDoubles());
    }

    @Test
    public void addSingle() {
        JavaArray a = array(new double[][]{
                {0, 1, 2},
                {3, 4, 5}
        });
        JavaArray b = new JavaArray(new double[]{10.0});

        JavaArray c = a.add(b);

        double[][] out = (double[][]) c.toDoubles();
        assertArrayEquals(new double[] {10, 11, 12}, out[0]);
        assertArrayEquals(new double[] {13, 14, 15}, out[1]);
    }

    @Test
    public void addRow() {
        JavaArray a = array(new double[][]{
                {0, 1, 2},
                {3, 4, 5}
        });
        JavaArray b = new JavaArray(new double[]{1, 10, 100});

        JavaArray c = a.add(b);

        double[][] out = (double[][]) c.toDoubles();
        assertArrayEquals(new double[] {1, 11, 102}, out[0]);
        assertArrayEquals(new double[] {4, 14, 105}, out[1]);
    }

    @Test
    public void add() {
        JavaArray a = array(new double[][]{
                {0, 1, 2},
                {3, 4, 5}
        });
        JavaArray b = range(6).reshape(2, 3);

        JavaArray c = a.add(b);

        double[][] out = (double[][]) c.toDoubles();
        assertArrayEquals(new double[] {0, 2, 4}, out[0]);
        assertArrayEquals(new double[] {6, 8, 10}, out[1]);
    }

}