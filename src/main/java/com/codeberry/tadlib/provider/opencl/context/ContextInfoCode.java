package com.codeberry.tadlib.provider.opencl.context;

import com.sun.jna.Native;
import com.sun.jna.Pointer;

public abstract class ContextInfoCode<R> {
    public static final ContextInfoCode<Integer> CL_CONTEXT_REFERENCE_COUNT = new ContextInfoCode<>(0x1080) {
        @Override
        Integer getValue(Pointer buf, long actualValueSize) {
            return buf.getInt(0);
        }
    };

    public static final ContextInfoCode<Pointer[]> CL_CONTEXT_DEVICES = new ContextInfoCode<>(0x1081) {
        @Override
        Pointer[] getValue(Pointer buf, long actualValueSize) {
            return buf.getPointerArray(0, (int) actualValueSize / Native.POINTER_SIZE);
        }
    };

    public static final ContextInfoCode<Pointer[]> CL_CONTEXT_PROPERTIES = new ContextInfoCode<>(0x1082) {
        @Override
        Pointer[] getValue(Pointer buf, long actualValueSize) {
            return buf.getPointerArray(0, (int) actualValueSize);
        }
    };

    public static final ContextInfoCode<Integer> CL_CONTEXT_NUM_DEVICES = new ContextInfoCode<>(0x1083) {
        @Override
        Integer getValue(Pointer buf, long actualValueSize) {
            return buf.getInt(0);
        }
    };

    public final int code;

    ContextInfoCode(int code) {
        this.code = code;
    }

    abstract R getValue(Pointer buf, long actualValueSize);

}