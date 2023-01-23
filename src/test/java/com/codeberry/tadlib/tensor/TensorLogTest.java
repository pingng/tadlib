package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.java.JavaProvider;
//import com.codeberry.tadlib.provider.opencl.OpenCLProvider;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static com.codeberry.tadlib.util.MatrixTestUtils.assertEqualsMatrix;

public class TensorLogTest {
    @BeforeEach
    public void init() {
//        ProviderStore.setProvider(new JavaProvider());
        ProviderStore.setProvider(new JavaProvider());
    }

    @Test
    public void log() {
        Tensor input = new Tensor(ProviderStore.array(new double[]{10, 1, 2, 3, 4, 5, 6, 7, 8}).reshape(1, 3, 3));

        Tensor log = Ops.log(input);
        log.backward(ProviderStore.array(new double[]{10., 2., 30.,
                4., 50, 6,
                70, 8, 90}).reshape(1, 3, 3));
//        log.backward();

        assertEqualsMatrix(ProviderStore.array(new double[]{
                        Math.log(10), Math.log(1), Math.log(2),
                        Math.log(3), Math.log(4), Math.log(5),
                        Math.log(6), Math.log(7), Math.log(8)
                }).reshape(1, 3, 3).toDoubles(),
                log.toDoubles());

        System.out.println(Arrays.deepToString((Object[]) input.grad().toDoubles()));
        assertEqualsMatrix(ProviderStore.array(new double[]{
                        1., 2., 15., 1.3333334, 12.5, 1.2, 11.666667, 1.1428572, 11.25
//                        1./10, 1./1, 1./2, 1./3, 1./4, 1./5, 1./6, 1./7, 1./8
                }).reshape(1, 3, 3).toDoubles(),
                input.grad().toDoubles());
    }
}
