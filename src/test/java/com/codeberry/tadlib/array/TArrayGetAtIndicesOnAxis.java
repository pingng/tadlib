package com.codeberry.tadlib.array;

import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.java.NDArray;
import com.codeberry.tadlib.provider.java.JavaProvider;
//import com.codeberry.tadlib.provider.opencl.OpenCLProvider;
import com.codeberry.tadlib.util.MatrixTestUtils;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static com.codeberry.tadlib.array.TArrayFactory.intRange;
import static org.junit.jupiter.api.Assertions.*;

public class TArrayGetAtIndicesOnAxis {

    @BeforeEach
    public void init() {
//        ProviderStore.setProvider(new JavaProvider());
        ProviderStore.setProvider(new JavaProvider());
    }

    @Test
    public void axisOutOfBound() {
        NDArray input = ProviderStore.array(new double[]{
                1, 2, 3, 5, 20, 2
        });

        NDIntArray indices = ProviderStore.array(3);

        assertThrows(AxisOutOfBounds.class, () ->
                input.getAtIndicesOnAxis(indices, 1));
        assertThrows(AxisOutOfBounds.class, () ->
                input.getAtIndicesOnAxis(indices, -2));
    }

    @Test
    public void dimensionMismatch() {
        //@formatter:off
        NDArray input = ProviderStore.array(new double[]{
                1, 2,3,   0, 3,100,
                6, 2,1,  -5, 4,  3,
                5,20,2,   1,-5, 20
        }).reshape(3, 2, 3);
        //@formatter:on

        NDIntArray indices = intRange(3 * 3)
                .reshape(3, 3);

        assertNotNull(input.getAtIndicesOnAxis(indices, 1));

        assertThrows(DimensionMismatch.class, () ->
                input.getAtIndicesOnAxis(intRange(3 * 3), 1));
        assertThrows(DimensionMismatch.class, () ->
                input.getAtIndicesOnAxis(intRange(3 * 2)
                        .reshape(3, 2), 1));
        assertThrows(DimensionMismatch.class, () ->
                input.getAtIndicesOnAxis(intRange(3 * 3)
                        .reshape(3, 3, 1), 1));
    }

    @Test
    public void getFrom1D() {
        //@formatter:off
        NDArray input = ProviderStore.array(new double[]{
                1, 2, 3, 5, 20, 2
        });
        //@formatter:on

        NDIntArray indices = ProviderStore.array(3);

        NDArray indexedVals = input.getAtIndicesOnAxis(indices, 0);
        System.out.println(indices.getShape());
        //@formatter:off
        //@formatter:on
        assertEquals(5.0, (double) indexedVals.toDoubles());
    }

    @Test
    public void getFrom2D() {
        //@formatter:off
        NDArray input = ProviderStore.array(new double[]{
                1, 2,3,
                6, 2,1,
                5,20,2,
        }).reshape(3, 3);
        //@formatter:on

        NDIntArray indices = ProviderStore.array(new int[] {
                2, 0, 1
        });

        NDArray indexedVals = input.getAtIndicesOnAxis(indices, 1);
        System.out.println(indices.getShape());
        //@formatter:off
        NDArray expected = ProviderStore.array(new double[]{
                3, 6, 20
        });
        //@formatter:on
        MatrixTestUtils.assertEqualsMatrix(expected.toDoubles(), indexedVals.toDoubles());
    }

    @Test
    public void getFrom3D() {
        //@formatter:off
        NDArray input = ProviderStore.array(new double[]{
                1, 2,3,   0, 3,100,
                6, 2,1,  -5, 4,  3,
                5,20,2,   1,-5, 20
        }).reshape(3, 2, 3);
        //@formatter:on

        NDIntArray indices = ProviderStore.array(new int[][] {
                {0, 1, 1},
                {0, 1, 1},
                {0, 0, 1},
        });

        NDArray indexedVals = input.getAtIndicesOnAxis(indices, 1);
        System.out.println(indices.getShape());
        //@formatter:off
        NDArray expected = ProviderStore.array(new double[]{
                1, 3, 100,
                6, 4,   3,
                5, 20, 20
        }).reshape(3, 3);
        //@formatter:on
        MatrixTestUtils.assertEqualsMatrix(expected.toDoubles(), indexedVals.toDoubles());
    }
}
