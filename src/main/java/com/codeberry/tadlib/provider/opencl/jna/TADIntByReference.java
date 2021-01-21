package com.codeberry.tadlib.provider.opencl.jna;

public class TADIntByReference extends TADByReference {

    public TADIntByReference() {
        this(0);
    }

    public TADIntByReference(int value) {
        super(4);
        setValue(value);
    }

    @Override
    protected Class<?> getDataType() {
        return Integer.class;
    }

    @Override
    protected String getStrValue() {
        return Integer.toString(getValue());
    }

    public void setValue(int value) {
        getPointer().setInt(0, value);
    }

    public int getValue() {
        return getPointer().getInt(0);
    }
}
