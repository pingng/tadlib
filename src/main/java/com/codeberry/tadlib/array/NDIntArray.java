package com.codeberry.tadlib.array;

import com.codeberry.tadlib.memorymanagement.DisposalRegister;
import com.codeberry.tadlib.provider.java.NDArray;

public interface NDIntArray extends DisposalRegister.Disposable {

    @Override
    default void prepareDependenciesForDisposal() {
        waitForValueReady();
    }

    default void waitForValueReady() {
        // do nothing
    }

    @Override
    default void dispose() {
        // do nothing
    }

    Object toInts();

    Shape getShape();

    int dataAt(int... indices);

    NDIntArray reshape(int... dims);

    NDArray compare(NDArray other, Comparison comparison, double trueValue, double falseValue);

    NDIntArray compare(NDIntArray other, Comparison comparison, int trueValue, int falseValue);
}
