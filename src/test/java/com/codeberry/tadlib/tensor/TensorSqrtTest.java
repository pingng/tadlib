package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.opencl.OpenCLProvider;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static com.codeberry.tadlib.util.MatrixTestUtils.*;

public class TensorSqrtTest {
    @BeforeEach
    public void init() {
//        ProviderStore.setProvider(new JavaProvider()); enableMultiThreading();
        ProviderStore.setProvider(new OpenCLProvider());
    }

    @Test
    public void sqrt() {
        Tensor input = new Tensor(ProviderStore.array(new double[]{10, 1, 2, 3, 4, 5, 6, 7, 8}).reshape(1, 3, 3));

        Tensor sqrt = Ops.sqrt(input);
        sqrt.backward(ProviderStore.array(new double[]{10., 2., 30.,
                4., 50, 6,
                70, 8, 90}).reshape(1, 3, 3));

        assertEqualsMatrix(ProviderStore.array(new double[]{
                        Math.sqrt(10), Math.sqrt(1), Math.sqrt(2),
                        Math.sqrt(3), Math.sqrt(4), Math.sqrt(5),
                        Math.sqrt(6), Math.sqrt(7), Math.sqrt(8)
                }).reshape(1, 3, 3).toDoubles(),
                sqrt.toDoubles());

        System.out.println(Arrays.deepToString((Object[]) input.getGradient().toDoubles()));
        assertEqualsMatrix(ProviderStore.array(new double[]{
                        1.5811388, 1.0000001, 10.606603, 1.1547005, 12.500002, 1.3416407,
                        14.288691, 1.511858, 15.909903
                }).reshape(1, 3, 3).toDoubles(),
                input.getGradient().toDoubles());
    }
}
