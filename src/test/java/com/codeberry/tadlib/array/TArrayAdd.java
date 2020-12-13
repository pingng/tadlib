package com.codeberry.tadlib.array;

import org.junit.jupiter.api.Test;

import static com.codeberry.tadlib.array.TArrayFactory.*;
import static org.junit.jupiter.api.Assertions.*;

class TArrayAdd {
    @Test
    public void plainValue() {
        TArray a = new TArray(2.0);
        TArray b = new TArray(6.0);

        TArray c = a.add(b);

        assertEquals(8.0, (double) c.toDoubles());
    }

    @Test
    public void addSingle() {
        TArray a = array(new double[][]{
                {0, 1, 2},
                {3, 4, 5}
        });
        TArray b = new TArray(new double[]{10.0});

        TArray c = a.add(b);

        double[][] out = (double[][]) c.toDoubles();
        assertArrayEquals(new double[] {10, 11, 12}, out[0]);
        assertArrayEquals(new double[] {13, 14, 15}, out[1]);
    }

    @Test
    public void addRow() {
        TArray a = array(new double[][]{
                {0, 1, 2},
                {3, 4, 5}
        });
        TArray b = new TArray(new double[]{1, 10, 100});

        TArray c = a.add(b);

        double[][] out = (double[][]) c.toDoubles();
        assertArrayEquals(new double[] {1, 11, 102}, out[0]);
        assertArrayEquals(new double[] {4, 14, 105}, out[1]);
    }

    @Test
    public void add() {
        TArray a = array(new double[][]{
                {0, 1, 2},
                {3, 4, 5}
        });
        TArray b = range(6).reshape(2, 3);

        TArray c = a.add(b);

        double[][] out = (double[][]) c.toDoubles();
        assertArrayEquals(new double[] {0, 2, 4}, out[0]);
        assertArrayEquals(new double[] {6, 8, 10}, out[1]);
    }

}