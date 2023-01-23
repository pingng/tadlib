package com.codeberry.tadlib.provider.java;

import com.codeberry.tadlib.array.CannotSqueezeNoneSingleDimension;
import com.codeberry.tadlib.array.DuplicatedSqueezeDimension;
import com.codeberry.tadlib.array.Shape;

import java.util.Arrays;
import java.util.BitSet;

import static java.lang.Math.min;

public class JavaShape implements Shape {
    protected final int[] dims;

    public final int size;
    public final int dimCount;

    private JavaShape() {
        this(new int[] { } );
    }

    public JavaShape(int... dims) {
//        System.out.println("JavaShape.JavaShape");
        this.dims = dims;
        this.dimCount = dims.length;
        this.size = mul(dims);
    }

    public static JavaShape of(int[] srcDims) {
        return new JavaShape(Arrays.copyOf(srcDims, srcDims.length));
    }

    public static final JavaShape zeroDim = new JavaShape();

    public static JavaShape shape(int... dims) {
        return new JavaShape(dims);
    }

//    public static JavaShape like(double[][] values) {
//        return new JavaShape(values.length, values[0].length);
//    }

    public int at(int dim) {
        if (dims.length == 0) return 0;
        return dims[wrapIndex(dim)];
    }

    protected int wrapIndex(int dim) {
        return dim >= 0 ? dim : dimCount + dim;
    }

    public JavaShape reshape(int[] dims) {
        return new JavaShape(reshapeDims(dims));
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
        int dimCount = -this.dimCount;
        for (int i = -1; i >= dimCount; i--) {
            int shapeDimSize = at(i);
            int index = min(indices[indices.length + i], shapeDimSize - 1);
            offset += blockSize * index;
            blockSize *= shapeDimSize;
        }
        return offset;
    }

    public JavaShape squeeze(int... removeSingleDimsIndices) {
        if (this instanceof ReorderedJavaShape) {
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
                throw new DuplicatedSqueezeDimension("dim=" + idx);
            }
        }
        for (int i = 0; i < _tmp.length; i++) {
            if (this.dims[_tmp[i]] != 1) {
                throw new CannotSqueezeNoneSingleDimension("index=" + i + " dim=" + _tmp[i]);
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
        return new JavaShape(dims);
    }

    // remove any reorderin etc.
    public JavaShape normalOrderedCopy() {
        return new JavaShape(Arrays.copyOf(this.dims, this.dims.length));
    }

    @Override
    public int[] getDimensions() {
        return dims;
    }

    public double[] convertDataToShape(double[] data, JavaShape tgtShape) {
        if (tgtShape.getClass() == this.getClass()) {
            return Arrays.copyOf(data, data.length);
        }

        double[] cp = new double[data.length];

        fillIntoDataArray(data, cp, tgtShape, newIndexArray(), 0);

        return cp;
    }

    private void fillIntoDataArray(double[] src, double[] tgt,
                                   JavaShape tgtShape, int[] indices, int dim) {
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
        return Shape.asString(this);
    }

    public JavaShape copy() {
        return new JavaShape(dims);
    }

    @Override
    public int getDimCount() {
        return dimCount;
    }

    @Override
    public long getSize() {
        return size;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof Shape) {
            return equalsShape((Shape) obj);
        }
        return false;
    }

    @Override
    public int hashCode() {
        return shapeHashCode();
    }
}
