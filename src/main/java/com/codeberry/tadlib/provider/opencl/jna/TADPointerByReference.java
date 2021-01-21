package com.codeberry.tadlib.provider.opencl.jna;

import com.sun.jna.Native;
import com.sun.jna.Pointer;

public class TADPointerByReference extends TADByReference {

    public TADPointerByReference() {
        this(null);
    }

    public TADPointerByReference(Pointer pointer) {
        super(Native.POINTER_SIZE);
        setValue(pointer);
    }

    public void setValue(Pointer value) {
        getPointer().setPointer(0, value);
    }

    public Pointer getValue() {
        return getPointer().getPointer(0);
    }

    @Override
    protected Class<?> getDataType() {
        return Pointer.class;
    }

    @Override
    protected String getStrValue() {
        Pointer p = getPointer();
        return (p == null ? "null" : p.toString());
    }
}
