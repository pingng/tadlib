package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.array.NDArray;

public interface GradFunc {
    NDArray calcGradient(NDArray gradient);
}
