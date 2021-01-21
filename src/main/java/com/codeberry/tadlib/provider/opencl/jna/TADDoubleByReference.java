package com.codeberry.tadlib.provider.opencl.jna;

public class TADDoubleByReference extends TADByReference {

    public TADDoubleByReference() {
        this(0.0);
    }

    public TADDoubleByReference(double value) {
        super(8);
        setValue(value);
    }

    public void setValue(double value) {
        getPointer().setDouble(0, value);
    }

    @Override
    protected Class<?> getDataType() {
        return double.class;
    }

    @Override
    protected String getStrValue() {
        return Double.toString(getValue());
    }

    public double getValue() {
        return getPointer().getDouble(0);
    }
}
