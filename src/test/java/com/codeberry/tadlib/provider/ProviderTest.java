package com.codeberry.tadlib.provider;

import com.codeberry.tadlib.array.NDIntArray;
import com.codeberry.tadlib.provider.java.JavaShape;
import com.codeberry.tadlib.array.NDArray;
import com.codeberry.tadlib.provider.java.JavaArray;
import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.provider.java.JavaProvider;
import com.codeberry.tadlib.provider.opencl.OclArray;
import com.codeberry.tadlib.provider.opencl.OpenCLProvider;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class ProviderTest {
    @Test
    public void javaProvider() {
        ProviderStore.setProvider(new JavaProvider());

        NDArray a = ProviderStore.array(0.);

        assertEquals(JavaArray.class, a.getClass());
    }

    @Test
    public void OpenCLProvider() {
        ProviderStore.setProvider(new OpenCLProvider());

        NDArray a = ProviderStore.array(0.);

        assertEquals(OclArray.class, a.getClass());
    }

    @Test
    public void dummyProvider() {
        ProviderStore.setProvider(new Provider() {
            @Override
            public NDArray createArray(double v) {
                return new DummyArray();
            }

            @Override
            public NDArray createArray(Object multiDimArray) {
                return new DummyArray();
            }

            @Override
            public NDIntArray createIntArray(Object multiDimArray) {
                return null;
            }

            @Override
            public NDIntArray createIntArray(int v) {
                return null;
            }

            @Override
            public NDIntArray createIntArrayWithValue(Shape shape, int v) {
                return null;
            }

            @Override
            public NDArray createArray(double[] data, Shape shape) {
                return new DummyArray();
            }

            @Override
            public Shape createShape(int... dims) {
                return new JavaShape(dims);
            }

            @Override
            public NDArray createArrayWithValue(Shape shape, double v) {
                return new DummyArray();
            }
        });

        NDArray a = ProviderStore.array(0.);

        assertEquals(DummyArray.class, a.getClass());
    }
}
