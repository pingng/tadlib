package com.codeberry.tadlib.array;

import com.codeberry.tadlib.provider.ProviderStore;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.IntFunction;

import static java.lang.Boolean.FALSE;

public interface Shape {
    int at(int dim);

    int getDimCount();

    long getSize();

    Shape reshape(int... dims);

    Shape normalOrderedCopy();

    int[] getDimensions();

    static String asString(Shape shape) {
        if (shape.getDimCount() == 0) {
            return "<>";
        }
        StringBuilder buf = new StringBuilder("<");
        for (int i = 0; i < shape.getDimCount(); i++) {
            buf.append(shape.at(i)).append(",");
        }
        buf.setLength(buf.length() - 1);
        buf.append(">");
        return buf.toString();
    }

    default int atOrDefault(int dim, int defaultValue) {
        int dimCount = getDimCount();
        int i = (dim >= 0 ? dim : dimCount + dim);
        if (i >= 0 && i < dimCount) {
            return at(i);
        }
        return defaultValue;
    }

    default int calcDataIndex(int... indices) {
        int idx = 0;
        int blockSize = 1;
        for (int i = indices.length - 1; i >= 0; i--) {
            idx += indices[i] * blockSize;
            blockSize *= at(i);
        }
        return idx;
    }

    default int[] newIndexArray() {
        return new int[getDimCount()];
    }

    default Object newValueArray(Class<?> componentType) {
        return Array.newInstance(componentType, toDimArray());
    }

    default Boolean[] newCollapseArray() {
        Boolean[] b = new Boolean[getDimCount()];
        Arrays.fill(b, FALSE);
        return b;
    }

    default int[] toDimArray() {
        int[] dims = new int[getDimCount()];
        for (int i = 0; i < dims.length; i++) {
            dims[i] = at(i);
        }
        return dims;
    }

    default boolean equalsShape(Shape other) {
        if (this == other) return true;
        if (other != null &&
                getSize() == other.getSize() && getDimCount() == other.getDimCount()) {
            for (int i = getDimCount() - 1; i >= 0; i--) {
                if (at(i) != other.at(i)) {
                    return false;
                }
            }
            return true;
        }
        return false;
    }

    default int shapeHashCode() {
        int result = Objects.hash(getSize(), getDimCount());
        for (int i = getDimCount() - 1; i >= 0; i--) {
            result = 31 * result + at(i);
        }
        return result;
    }

    default int[] reshapeDims(int... dims) {
        long restSize = this.getSize();
        int deduceDim = -1;

        for (int i = 0; i < dims.length; i++) {
            int dim = dims[i];
            if (dim > 0) {
                if ((restSize % dim) != 0) {
                    throw new InvalidTargetShape("Not divisible: Index: " + i + " dim: " + dim);
                }
                restSize /= dim;
            } else if (dim == -1) {
                if (deduceDim != -1) {
                    throw new InvalidTargetShape("Multiple -1 dim not allowed: Index: " + i);
                }
                deduceDim = i;
            } else {
                throw new InvalidTargetShape("Invalid dim: Index: " + i + " dim:" + dim);
            }
        }

        if (deduceDim != -1) {
            dims[deduceDim] = Math.toIntExact(restSize);
        }

        return dims;
    }

    default  <A> A[] forEachDim(Function<Integer, A> mapper, IntFunction<A[]> returnFactory) {
        A[] data = returnFactory.apply(getDimCount());
        for (int i = 0; i < data.length; i++) {
            A v = mapper.apply(at(i));
            data[i] = v;
        }
        return data;
    }

    default boolean hasSingleDims() {
        for (int i = 0; i < getDimCount(); i++)
            if (at(i) == 1)
                return true;
        return false;
    }

    default boolean correspondsTo(Shape other) {
        int dimCount = this.getDimCount();

        if (dimCount == other.getDimCount()) {
            for (int i = 0; i < dimCount; i++) {
                if (at(i) != other.at(i)) {
                    return false;
                }
            }
            return true;
        }
        return false;
    }

    default Shape withDimAt(int index, int dimLength) {
        int[] dims = toDimArray();
        int i = (index >= 0 ? index : getDimCount() + index);
        dims[i] = dimLength;
        return ProviderStore.shape(dims);
    }

    default long mulDims(int fromDim, int toDimExclusive) {
        int f = (fromDim >= 0 ? fromDim : getDimCount() + fromDim);
        int t = (toDimExclusive >= 0 ? toDimExclusive : getDimCount() + toDimExclusive);
        if (f < t) {
            int r = 1;
            for (int i = f; i < t; i++) {
                r *= at(i);
            }
            return r;
        }
        return 0;
    }

    default boolean isValid(int[] indices) {
        for (int i = indices.length - 1; i >= 0; i--) {
            int index = indices[i];
            if (index <= -1 || index >= at(i))
                return false;
        }
        return true;
    }

    default Shape removeDimAt(int axis) {
        int dimCount = getDimCount();
        int[] result = new int[dimCount - 1];

        int _axis = (axis >= 0 ? axis : dimCount + axis);
        int index = 0;
        for (int i = 0; i < dimCount; i++) {
            if (_axis != i) {
                result[index++] = at(i);
            }
        }

        return ProviderStore.shape(result);
    }

    default Shape appendDim(int lastDimLength) {
        int dimCount = getDimCount();
        int[] result = Arrays.copyOf(toDimArray(), dimCount + 1);
        result[dimCount] = lastDimLength;
        return ProviderStore.shape(result);
    }

    default int wrapNegIndex(int dimIndex) {
        return (dimIndex >= 0 ? dimIndex : getDimCount() + dimIndex);
    }
}
