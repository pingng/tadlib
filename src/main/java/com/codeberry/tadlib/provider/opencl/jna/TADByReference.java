package com.codeberry.tadlib.provider.opencl.jna;

import com.sun.jna.PointerType;

public abstract class TADByReference extends PointerType {
    public TADByReference(int byteSize) {
        super(new TADMemory(byteSize));
    }

    public void dispose() {
        ((TADMemory) super.getPointer()).dispose();
    }

    protected abstract Class<?> getDataType();
    protected abstract String getStrValue();

    @Override
    public String toString() {
        return "{ByRef:" + getClass().getSimpleName() +
                ",dType=" + getDataType().getSimpleName() +
                ",value=" + getStrValue() + "}";
    }
}
