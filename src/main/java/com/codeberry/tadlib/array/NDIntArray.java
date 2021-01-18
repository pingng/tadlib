package com.codeberry.tadlib.array;

import com.codeberry.tadlib.memorymanagement.DisposalRegister;

import java.util.List;

import static java.util.Collections.singletonList;

public interface NDIntArray extends DisposalRegister.Disposable, DisposalRegister.DisposableContainer<NDIntArray> {

    @Override
    default void prepareDependenciesForDisposal() {
        waitForValueReady();
    }

    default void waitForValueReady() {
        // do nothing
    }

    @Override
    default List<NDIntArray> getDisposables() {
        return singletonList(this);
    }

    @Override
    default void dispose() {
        // do nothing
    }

    Object toInts();

    Shape getShape();

    int dataAt(int... indices);

    NDIntArray reshape(int... dims);
}
