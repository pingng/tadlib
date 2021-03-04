package com.codeberry.tadlib.provider;

import com.codeberry.tadlib.array.NDArray;
import com.codeberry.tadlib.array.NDIntArray;
import com.codeberry.tadlib.array.Shape;

public interface Provider {
    NDArray createArray(double v);
    NDArray createArray(Object multiDimArray);

    NDIntArray createIntArray(Object multiDimArray);

    NDIntArray createIntArray(int v);

    NDIntArray createIntArrayWithValue(Shape shape, int v);

    NDArray createArray(double[] data, Shape shape);

    Shape createShape(int... dims);

    NDArray createArrayWithValue(Shape shape, double v);

    String getShortDescription();
}
