package com.codeberry.tadlib.provider;

import com.codeberry.tadlib.array.NDArray;

public interface Provider {
    NDArray createArray(double v);
}
