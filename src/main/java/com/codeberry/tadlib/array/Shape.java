package com.codeberry.tadlib.array;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.BitSet;
import java.util.function.Function;
import java.util.function.IntFunction;

import static java.lang.Boolean.FALSE;
import static java.lang.Math.min;

public class Shape {
    protected final int[] dims;

    public final int size;
    public final int dimCount;

    public Shape(int... dims) {
        this.dims = dims;
        this.dimCount = dims.length;
        this.size = mul(dims);
    }

    public static Shape of(int[] srcDims) {
        return new Shape(Arrays.copyOf(srcDims, srcDims.length));
    }

    public static Shape zeroDim() {
        return new Shape();
    }

    public static Shape shape(int... dims) {
        return new Shape(dims);
    }

    public <A> A[] forEachDim(Function<Integer, A> mapper, IntFunction<A[]> returnFactory) {
        A[] data = returnFactory.apply(dims.length);
        for (int i = 0; i < dims.length; i++) {
            A v = mapper.apply(at(i));
            data[i] = v;
        }
        return data;
    }

    public static Shape like(double[][] values) {
        return new Shape(values.length, values[0].length);
    }

    public int[] newIndexArray() {
        return new int[dims.length];
    }

    public Object newValueArray() {
        return Array.newInstance(double.class, dims);
    }

    public int at(int dim) {
        return dims[wrapIndex(dim)];
    }

    public int atOrNeg(int dim) {
        int i = wrapIndex(dim);
        if (i >= 0 && i < dimCount) {
            return dims[i];
        }
        return -1;
    }

    protected int wrapIndex(int dim) {
        return dim >= 0 ? dim : dimCount + dim;
    }

    public int calcDataIndex(int... indices) {
        int idx = 0;
        int blockSize = 1;
        for (int i = indices.length - 1; i >= 0; i--) {
            idx += indices[i] * blockSize;
            blockSize *= dims[i];
        }
        return idx;
    }

    public Shape reshape(int[] dims) {
        int restSize = this.size;
        int deduceDim = -1;

        for (int i = 0; i < dims.length; i++) {
            int dim = dims[i];
            if (dim > 0) {
                if ((restSize % dim) != 0) {
                    throw new TArray.InvalidTargetShape("Not divisible: Index: " + i + " dim: " + dim);
                }
                restSize /= dim;
            } else if (dim == -1) {
                if (deduceDim != -1) {
                    throw new TArray.InvalidTargetShape("Multiple -1 dim not allowed: Index: " + i);
                }
                deduceDim = i;
            } else {
                throw new TArray.InvalidTargetShape("Invalid dim: Index: " + i + " dim:" + dim);
            }
        }

        if (deduceDim != -1) {
            dims[deduceDim] = restSize;
        }

        return new Shape(dims);
    }

    private static int mul(int[] dims) {
        int targetSize = 1;
        for (int dim : dims) {
            targetSize *= dim;
        }
        return targetSize;
    }

    int getBroadcastOffset(int[] indices) {
        int offset = 0;
        int blockSize = 1;
        int dimCount = this.dimCount;
        for (int i = -1; i >= -dimCount; i--) {
            int shapeDimSize = at(i);
            int index = min(indices[indices.length + i], shapeDimSize - 1);
            offset += blockSize * index;
            blockSize *= shapeDimSize;
        }
        return offset;
    }

    public boolean hasSingleDims() {
        for (int dim : dims)
            if (dim == 1)
                return true;
        return false;
    }

    public Shape squeeze(int... removeSingleDimsIndices) {
        if (this instanceof ReorderedShape) {
            throw new UnsupportedOperationException("Reordered not supported");
        }
        int[] _tmp = Arrays.copyOf(removeSingleDimsIndices, removeSingleDimsIndices.length);
        for (int i = 0; i < _tmp.length; i++) {
            if (_tmp[i] <= -1) {
                _tmp[i] += dimCount;
            }
        }
        BitSet bitSet = new BitSet();
        for (int idx : _tmp) {
            if (!bitSet.get(idx)) {
                bitSet.set(idx);
            } else {
                throw new TArray.DuplicatedSqueezeDimension("dim=" + idx);
            }
        }
        for (int i = 0; i < _tmp.length; i++) {
            if (this.dims[_tmp[i]] != 1) {
                throw new TArray.CannotSqueezeNoneSingleDimension("index=" + i + " dim=" + _tmp[i]);
            }
        }
        int[] dims = new int[dimCount - _tmp.length];
        int idx = 0;
        for (int i = 0; i < this.dims.length; i++) {
            if (!bitSet.get(i)) {
                dims[idx] = this.dims[i];
                idx++;
            }
        }
        return new Shape(dims);
    }

    // remove any reorderin etc.
    public Shape normalOrderedCopy() {
        return new Shape(Arrays.copyOf(this.dims, this.dims.length));
    }

    public double[] convertDataToShape(double[] data, Shape tgtShape) {
        if (tgtShape.getClass() == this.getClass()) {
            return Arrays.copyOf(data, data.length);
        }

        double[] cp = new double[data.length];

        fillIntoDataArray(data, cp, tgtShape, newIndexArray(), 0);

        return cp;
    }

    private void fillIntoDataArray(double[] src, double[] tgt,
                                   Shape tgtShape, int[] indices, int dim) {
        int len = at(dim);
        if (dim == dimCount - 1) {
            //...last index
            for (int i = 0; i < len; i++) {
                indices[dim] = i;
                int srcOffset = calcDataIndex(indices);
                int tgtOffset = tgtShape.calcDataIndex(indices);
                tgt[tgtOffset] =
                        src[srcOffset];
            }
        } else {
            for (int i = 0; i < len; i++) {
                indices[dim] = i;
                fillIntoDataArray(src, tgt, tgtShape, indices, dim + 1);
            }
        }
    }

    @Override
    public String toString() {
        if (dimCount == 0) {
            return "<>";
        }
        StringBuilder buf = new StringBuilder("<");
        for (int i = 0; i < dimCount; i++) {
            buf.append(at(i)).append(",");
        }
        buf.setLength(buf.length() - 1);
        buf.append(">");
        return buf.toString();
    }

    public Shape copy() {
        return new Shape(dims);
    }

    public double[] toDataArray(Object array) {
        double[] data = new double[size];
        fillIntoDataArray(array, data, newIndexArray(), 0);
        return data;
    }

    private void fillIntoDataArray(Object array, double[] data, int[] indices, int dim) {
        int len = Array.getLength(array);
        if (dim == dimCount - 1) {
            int tgtOffset = calcDataIndex(indices);
            double[] arr = (double[]) array;
            System.arraycopy(arr, 0, data, tgtOffset, len);
        } else {
            for (int i = 0; i < len; i++) {
                indices[dim] = i;
                fillIntoDataArray(Array.get(array, i), data, indices, dim + 1);
            }
        }
    }

    public int[] toDimArray() {
        int[] dims = new int[dimCount];
        for (int i = 0; i < dims.length; i++) {
            dims[i] = at(i);
        }
        return dims;
    }

    public boolean isValid(int[] indices) {
        for (int i = indices.length - 1; i >= 0; i--) {
            int index = indices[i];
            if (index <= -1 || index >= dims[i])
                return false;
        }
        return true;
    }

    public boolean correspondsTo(Shape other) {
        if (this.dimCount == other.dimCount) {
            for (int i = 0; i < dimCount; i++) {
                if (at(i) != other.at(i)) {
                    return false;
                }
            }
            return true;
        }
        return false;
    }

    public Boolean[] newCollapseArray() {
        Boolean[] b = new Boolean[dimCount];
        Arrays.fill(b, FALSE);
        return b;
    }

    public Shape withDimAt(int index, int dimLength) {
        int[] dims = toDimArray();
        dims[wrapIndex(index)] = dimLength;
        return new Shape(dims);
    }
}
