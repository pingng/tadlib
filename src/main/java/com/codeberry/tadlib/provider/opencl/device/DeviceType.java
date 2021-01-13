package com.codeberry.tadlib.provider.opencl.device;

import com.sun.jna.NativeLong;

public enum DeviceType {
    /* cl_device_type - bitfield */
    CL_DEVICE_TYPE_DEFAULT(1 << 0),
    CL_DEVICE_TYPE_CPU(1 << 1),
    CL_DEVICE_TYPE_GPU(1 << 2),
    CL_DEVICE_TYPE_ACCELERATOR(1 << 3),
    CL_DEVICE_TYPE_CUSTOM(1 << 4),
    CL_DEVICE_TYPE_ALL(0xFFFFFFFF);

    public final long bits;

    DeviceType(long bits) {
        this.bits = bits;
    }

    public static NativeLong toBits(DeviceType... types) {
        if (types.length == 0) {
            throw new IllegalArgumentException("Expected at least one type");
        }
        long bits = 0;
        for (DeviceType type : types) {
            bits |= type.bits;
        }
        return new NativeLong(bits, true);
    }

    public static DeviceType fromBits(long bits) {
        for (DeviceType type : values()) {
            if (type.bits == bits) {
                return type;
            }
        }
        throw new IllegalArgumentException("Unknown bits: " + Long.toBinaryString(bits));
    }
}
