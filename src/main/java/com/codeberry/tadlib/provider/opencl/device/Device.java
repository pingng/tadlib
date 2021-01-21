package com.codeberry.tadlib.provider.opencl.device;

import com.codeberry.tadlib.provider.opencl.OpenCL;
import com.codeberry.tadlib.provider.opencl.SizeT;
import com.sun.jna.Pointer;

import java.util.Arrays;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;

public class Device extends Pointer {
    private final int index;
    public final Info info;
    private final DevicesPointer parent;

    private Device(int index, Info info, DevicesPointer parent) {
        super(Pointer.nativeValue(parent.getDevicePointer(index)));
        this.index = index;
        this.info = info;
        this.parent = parent;
    }

    public long getWorkItemSize(int dimension) {
        return info.workItemSizes[dimension].longValue();
    }

    public static Device fromDevicesIndex(DevicesPointer parent, int index) {
        Device.Info info = parent.getDeviceInfo(parent, index);
        return new Device(index, info, parent);
    }

    public static <R> R mapDevicePointers(Function<OpenCL.PointerArray, R> mapper, List<Device> devices) {
        return OpenCL.PointerArray.mapPointerArray(mapper, devices.toArray(Pointer[]::new));
    }

    public static void useDevicePointers(Consumer<OpenCL.PointerArray> mapper, List<Device> devices) {
        OpenCL.PointerArray.usePointerArray(mapper, devices.toArray(Pointer[]::new));
    }

    @Override
    public String toString() {
        return "Device{" +
                "info=" + info +
                '}';
    }

    public int getMaxWorkGroupSize() {
        return (int) (Math.min(info.workItemSizes[0].longValue(), info.maxWorkGroupSize.longValue()));
    }

    public static class Info {
        public final String name;
        public final int maxComputeUnits;
        public final long doubleFpConfig;
        public final long globalMemSize;
        public final SizeT maxWorkGroupSize;
        public final SizeT[] workItemSizes;
        public final DeviceType type;

        public Info(String name, int maxComputeUnits, long doubleFpConfig, long globalMemSize, SizeT maxWorkGroupSize, SizeT[] workItemSizes, DeviceType type) {
            this.name = name;
            this.maxComputeUnits = maxComputeUnits;
            this.doubleFpConfig = doubleFpConfig;
            this.globalMemSize = globalMemSize;
            this.maxWorkGroupSize = maxWorkGroupSize;
            this.workItemSizes = workItemSizes;
            this.type = type;
        }

        @Override
        public String toString() {
            return "{" +
                    "name='" + name + '\'' +
                    ", maxComputeUnits=" + maxComputeUnits +
                    ", doubleFpConfig=" + doubleFpConfig +
                    ", globalMemSize=" + globalMemSize +
                    ", maxWorkGroupSize=" + maxWorkGroupSize +
                    ", workItemSizes=" + Arrays.toString(workItemSizes) +
                    ", type=" + type +
                    '}';
        }
    }
}
