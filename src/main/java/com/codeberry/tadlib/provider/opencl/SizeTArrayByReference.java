package com.codeberry.tadlib.provider.opencl;

import com.codeberry.tadlib.provider.opencl.jna.TADByReference;
import com.sun.jna.Pointer;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

import static com.sun.jna.Native.SIZE_T_SIZE;
import static java.util.stream.Collectors.*;

public class SizeTArrayByReference extends TADByReference {
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

    @Override
    protected Class<?> getDataType() {
        return SizeT[].class;
    }

    @Override
    protected String getStrValue() {
        List<Long> vals = new ArrayList<>();
        for (int i = 0; i < length; i++) {
            vals.add(getValue(i));
        }
        return vals.toString();
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
}
