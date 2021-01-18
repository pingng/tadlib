package com.codeberry.tadlib.array;

import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.java.JavaProvider;
import com.codeberry.tadlib.provider.opencl.OpenCLProvider;
import com.codeberry.tadlib.util.MatrixTestUtils;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class TArrayUpdateAtIndicesOnAxis {

    @BeforeEach
    public void init() {
        ProviderStore.setProvider(new JavaProvider());
//        ProviderStore.setProvider(new OpenCLProvider());
    }

    @Test
    public void axisOutOfBound_ForAxis1() {
        NDArray input = TArrayFactory.range(3 * 2 * 4).reshape(3, 2, 4);

        NDIntArray indices = TArrayFactory.intZerosShaped(3, 4);

        NDArray change = TArrayFactory.range(3 * 4).reshape(3, 4);

        assertNotNull(input.withUpdateAtIndicesOnAxis(indices, 1, change));

        assertThrows(AxisOutOfBounds.class, () -> input.withUpdateAtIndicesOnAxis(indices, 3, change));
        assertThrows(AxisOutOfBounds.class, () -> input.withUpdateAtIndicesOnAxis(indices, -4, change));
    }


    @Test
    public void update1d() {
        //@formatter:off
        NDArray input = ProviderStore.array(new double[]{
                1, 2, 3, 0, 3,100
        });
        //@formatter:on

        NDIntArray indices = ProviderStore.array(3);

        //@formatter:off
        NDArray change = ProviderStore.array(123.5);
        //@formatter:on

        NDArray output = input.withUpdateAtIndicesOnAxis(indices, 0, change);

        //@formatter:off
        NDArray expected = ProviderStore.array(new double[]{
                1, 2, 3, 123.5, 3,100
        });
        //@formatter:on

        MatrixTestUtils.assertEqualsMatrix(expected.toDoubles(), output.toDoubles());
    }

    @Test
    public void update3d() {
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

        //@formatter:off
        NDArray change = ProviderStore.array(new double[]{
                0.1, 0.2, 0.3,
                0.4, 0.5, 0.6,
                0.7, 0.8, 0.9
        }).reshape(3, 3);
        //@formatter:on

        NDArray output = input.withUpdateAtIndicesOnAxis(indices, 1, change);

        //@formatter:off
        NDArray expected = ProviderStore.array(new double[]{
                0.1, 2,3,   0, 0.2,0.3,
                0.4, 2,1,  -5, 0.5,0.6,
                0.7,0.8,2,   1,-5, 0.9
        }).reshape(3, 2, 3);
        //@formatter:on

        MatrixTestUtils.assertEqualsMatrix(expected.toDoubles(), output.toDoubles());
    }
}
