package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.array.TArray;

public interface GradFunc {
    TArray calcGradient(TArray gradient);
}
