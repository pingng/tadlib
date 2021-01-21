package com.codeberry.tadlib.provider.opencl.jna;

public class TADLongByReference extends TADByReference {

    public TADLongByReference() {
        this(0L);
    }

    public TADLongByReference(long value) {
        super(8);
        setValue(value);
    }

    public void setValue(long value) {
        getPointer().setLong(0, value);
    }

    public long getValue() {
        return getPointer().getLong(0);
    }

    @Override
    protected Class<?> getDataType() {
        return Long.class;
    }

    @Override
    protected String getStrValue() {
        return Long.toString(getValue());
    }
}
