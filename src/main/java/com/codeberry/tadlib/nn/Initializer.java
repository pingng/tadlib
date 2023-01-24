package com.codeberry.tadlib.nn;

import com.codeberry.tadlib.provider.java.NDArray;
import com.codeberry.tadlib.tensor.Tensor;

import java.util.List;
import java.util.Random;

public interface Initializer {
    public void initialize(List<Tensor> initializables);

    public static class UniformInitializer implements Initializer {

        private final Random rng;
        private final float radius;

        public UniformInitializer(Random rng, float radius) {
            this.rng = rng;
            this.radius = radius;
        }

        @Override
        public void initialize(List<Tensor> initializables) {
            for (Tensor t : initializables) {
                NDArray z = new NDArray(t.shape());
                for (int i = 0; i < z.data.length; i++)
                    z.data[i] = rng.nextDouble(-radius, +radius);
                t.set(z);
            }
        }
    }
}
