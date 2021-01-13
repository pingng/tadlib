package com.codeberry.tadlib.provider.opencl;

import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.ByReference;

public class SizeTByReference extends ByReference {
    public SizeTByReference() {
        this(0);
    }

    public SizeTByReference(long v) {
        super(Native.SIZE_T_SIZE);
        setValue(v);
    }

    public void setValue(long value) {
        getPointer().setLong(0, value);
    }

    public long getValue() {
        return getPointer().getInt(0);
    }

    @Override
    public String toString() {
        return String.format("size_t@0x%1$x=0x%2$x (%2$d)", Pointer.nativeValue(getPointer()), getValue());
    }
}
