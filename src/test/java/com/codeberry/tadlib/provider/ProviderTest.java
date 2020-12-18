package com.codeberry.tadlib.provider;

import com.codeberry.tadlib.array.NDArray;
import com.codeberry.tadlib.array.JavaArray;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class ProviderTest {
    @Test
    public void javaProvider() {
        ProviderStore.setProvider(new JavaProvider());

        NDArray a = ProviderStore.array(0);

        Assertions.assertEquals(JavaArray.class, a.getClass());
    }
}
