package com.codeberry.tadlib.provider;

import com.codeberry.tadlib.provider.java.JavaIntArray;
import com.codeberry.tadlib.provider.java.NDArray;
import com.codeberry.tadlib.provider.java.Shape;

public interface Provider {
    NDArray createArray(double v);

    NDArray createArray(Object multiDimArray);

    JavaIntArray createIntArray(Object multiDimArray);

    JavaIntArray createIntArray(int v);

    JavaIntArray createIntArrayWithValue(Shape shape, int v);

    NDArray createArray(double[] data, Shape shape);

    Shape createShape(int... dims);

    NDArray createArrayWithValue(Shape shape, double v);

    String getShortDescription();
}
