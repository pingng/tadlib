package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.array.JavaArray;

public interface GradFunc {
    JavaArray calcGradient(JavaArray gradient);
}
