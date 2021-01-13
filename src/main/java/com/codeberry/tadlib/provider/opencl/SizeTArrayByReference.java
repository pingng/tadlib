package com.codeberry.tadlib.provider.opencl;

import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.ByReference;

import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static com.sun.jna.Native.SIZE_T_SIZE;
import static java.util.stream.Collectors.*;

public class SizeTArrayByReference extends ByReference {
    private final int length;

    /** Default constructor meant for JNA */
    public SizeTArrayByReference() {
        this(1);
    }

    public SizeTArrayByReference(int length) {
        super(SIZE_T_SIZE * length);
        this.length = length;
        for (int i = 0; i < length; i++) {
            setValue(i, 0);
        }
    }

    public static SizeTArrayByReference toSizeTArray(long... values) {
        SizeTArrayByReference r = new SizeTArrayByReference(values.length);
        for (int i = 0; i < values.length; i++) {
            r.setValue(i, values[i]);
        }
        return r;
    }

    public int getLength() {
        return length;
    }

    public void setValue(int index, long value) {
        getPointer().setLong((long) SIZE_T_SIZE * index, value);
    }

    public long getValue(int index) {
        return getPointer().getLong((long) SIZE_T_SIZE * index);
    }

    @Override
    public String toString() {
        String vals = IntStream.range(0, length)
                .mapToLong(this::getValue)
                .mapToObj(Long::toString)
                .collect(joining(", "));
        return String.format("size_t@0x%1$x=[%2$s]", Pointer.nativeValue(getPointer()), vals);
    }
}
