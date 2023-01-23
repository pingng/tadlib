package com.codeberry.tadlib.array;

import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.java.NDArray;
import com.codeberry.tadlib.provider.java.JavaProvider;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import static com.codeberry.tadlib.array.TArrayFactory.*;
import static com.codeberry.tadlib.provider.ProviderStore.*;
import static org.junit.jupiter.api.Assertions.*;

class TArrayAdd {

    @BeforeEach
    public void init() {
        ProviderStore.setProvider(new JavaProvider());
        //setProvider(new OpenCLProvider());
    }

    @Test
    public void plainValue() {
        NDArray a = array(2.0);
        NDArray b = array(6.0);

        NDArray c = a.add(b);

        assertEquals(8.0, (double) c.toDoubles());
    }

    @Disabled
    @Test
    public void memoryIsGCedCorrectly_MeantForOpenCL() {
        double[] vals = rangeDoubles(100000);
        NDArray a = array(vals);
        NDArray b = array(6.0);

        NDArray c = a.add(b);

        for (int i = 0; i < 40000; i++) {
            NDArray old = c;
            c = old
                    .add(a)
                    .add(a);
            c.waitForValueReady();
//            System.out.println("i = " + i);
//            ((OclArray)old).dispose();
//            System.gc();
        }

        System.out.println(c);
    }

    @Test
    public void addSingle() {
        NDArray a = ProviderStore.array(new double[][]{
                {0, 1, 2},
                {3, 4, 5}
        });
        NDArray b = array(new double[]{10.0});

        NDArray c = a.add(b);

        double[][] out = (double[][]) c.toDoubles();
        assertArrayEquals(new double[]{10, 11, 12}, out[0]);
        assertArrayEquals(new double[]{13, 14, 15}, out[1]);
    }

    @Test
    public void addRow() {
        NDArray a = ProviderStore.array(new double[][]{
                {0, 1, 2},
                {3, 4, 5}
        });
        NDArray b = array(new double[]{1, 10, 100});

        NDArray c = a.add(b);

        double[][] out = (double[][]) c.toDoubles();
        assertArrayEquals(new double[]{1, 11, 102}, out[0]);
        assertArrayEquals(new double[]{4, 14, 105}, out[1]);
    }

    @Test
    public void addColumn() {
        NDArray a = ProviderStore.array(new double[][]{
                {0, 1, 2},
                {3, 4, 5}
        });
        NDArray b = ProviderStore.array(new double[][]{
                {1},
                {10},
        });

        NDArray c = a.add(b);

        double[][] out = (double[][]) c.toDoubles();
        assertArrayEquals(new double[]{1, 2, 3}, out[0]);
        assertArrayEquals(new double[]{13, 14, 15}, out[1]);
    }

    @Test
    public void addSequence() {
        NDArray a = ProviderStore.array(new double[][]{
                {0, 1, 2},
                {3, 4, 5}
        });
        NDArray b = ProviderStore.array(new double[][]{
                {1},
                {10},
        });

        NDArray c = a.add(b);
        NDArray d = c.add(ProviderStore.array(new double[][]{
                {100},
        }));

        double[][] out = (double[][]) d.toDoubles();
        assertArrayEquals(new double[]{101, 102, 103}, out[0]);
        assertArrayEquals(new double[]{113, 114, 115}, out[1]);
    }

    @Test
    public void addSameShape() {
        NDArray a = ProviderStore.array(new double[][]{
                {0, 1, 2},
                {3, 4, 5}
        });
        NDArray b = ProviderStore.array(new double[][]{
                {0, 1, 2},
                {3, 4, 10}
        });
        ;
        assertEquals(a.shape, b.shape);

        NDArray c = a.add(b);

        double[][] out = (double[][]) c.toDoubles();
        assertArrayEquals(new double[]{0, 2, 4}, out[0]);
        assertArrayEquals(new double[]{6, 8, 15}, out[1]);
    }

}