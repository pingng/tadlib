package com.codeberry.tadlib.provider.java;

import com.codeberry.tadlib.array.Shape;

import java.util.function.IntFunction;

import static com.codeberry.tadlib.array.util.DimensionUtils.*;
import static java.lang.Math.toIntExact;

public class CompareHelper {
    public interface CompareWriter<E, R> {
        R scalar(E value);

        R toArray(JavaShape shape);

        void prepareDate(int size);

        void write(int offset, E outVal);
    }

    public interface ValueComparator<E> {
        boolean compare(E leftV, E rightV);
    }

    static <E, R> R compare(ValueComparator<E> comparator, E trueValue, E falseValue,
                            Shape leftShape, Shape rightShape,
                            IntFunction<E> left, IntFunction<E> right,
                            CompareWriter<E, R> writer) {
        if (leftShape.getDimCount() == 0 &&
                rightShape.getDimCount() == 0) {

            E outVal = comparator.compare(left.apply(0), right.apply(0)) ?
                    trueValue : falseValue;

            return writer.scalar(outVal);
        }

        validateBroadcastShapes(leftShape, rightShape, -1);
        JavaShape outShape = NDArray.evalBroadcastOutputShape(leftShape, rightShape);

        writer.prepareDate(toIntExact(outShape.getSize()));

        int[] thisBroadcastBlockSizes = calcBroadcastBlockSizes(leftShape, outShape.getDimCount());
        int[] otherBroadcastBlockSizes = calcBroadcastBlockSizes(rightShape, outShape.getDimCount());

        fillCompare(comparator, trueValue, falseValue,
                left, thisBroadcastBlockSizes,
                right, otherBroadcastBlockSizes,
                writer, outShape, outShape.newIndexArray(), 0);

        return writer.toArray(outShape);
    }

    static <E, R> void fillCompare(ValueComparator<E> comparator, E trueValue, E falseValue,
                                   IntFunction<E> left, int[] leftBroadcastBlockSizes,
                                   IntFunction<E> right, int[] rightBroadcastBlockSizes,
                                   CompareWriter<E, R> writer, Shape outShape, int[] outIndices, int outDim) {
        int dimLen = outShape.at(outDim);
        for (int i = 0; i < dimLen; i++) {
            outIndices[outDim] = i;

            if (outDim == outIndices.length - 1) {
                //... then is last dimension
                E leftVal = left.apply(calcOffset(outIndices, leftBroadcastBlockSizes));
                E rightVal = right.apply(calcOffset(outIndices, rightBroadcastBlockSizes));

                E outVal = comparator.compare(leftVal, rightVal) ?
                        trueValue : falseValue;

                writer.write(outShape.calcDataIndex(outIndices), outVal);
            } else {
                fillCompare(comparator, trueValue, falseValue,
                        left, leftBroadcastBlockSizes,
                        right, rightBroadcastBlockSizes,
                        writer, outShape, outIndices, outDim + 1);
            }
        }
    }
}
