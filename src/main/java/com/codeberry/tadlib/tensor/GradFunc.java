package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.provider.java.NDArray;

public interface GradFunc {
    NDArray calcGradient(NDArray gradient);
}
