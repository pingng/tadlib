package com.codeberry.tadlib.provider.opencl;

import com.sun.jna.IntegerType;
import com.sun.jna.Native;
import com.sun.jna.Pointer;

// From: https://github.com/java-native-access/jna/issues/191
public class SizeT extends IntegerType {

    public static final SizeT POINTER_SIZE_T = new SizeT(Native.POINTER_SIZE);

    public SizeT() {
        this(0);
    }

    public SizeT(long value) {
        super(Native.SIZE_T_SIZE, value, true);
    }

    public static SizeT getSizeT(Pointer _buf, long sizeTIndex) {
        long offset = sizeTIndex * Native.SIZE_T_SIZE;

        long l = 0;
        for (int i = 0; i < Native.SIZE_T_SIZE; i++) {
            long _b = _buf.getByte(offset + i) & 0xFF;
            l |= _b << (i * 8);
        }

        return new SizeT(l);
    }
}
