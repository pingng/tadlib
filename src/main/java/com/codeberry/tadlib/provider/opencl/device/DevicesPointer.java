package com.codeberry.tadlib.provider.opencl.device;

import com.codeberry.tadlib.provider.opencl.OpenCLHelper;
import com.codeberry.tadlib.provider.opencl.SizeT;
import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Pointer;

public class DevicesPointer extends Memory {
    private final int count;

    public DevicesPointer(int count) {
        super((long) Native.POINTER_SIZE * count);
        this.count = count;
    }

    int getCount() {
        return count;
    }

    Pointer getDevicePointer(int index) {
        return getPointer((long) index * Native.POINTER_SIZE);
    }

    Device.Info getDeviceInfo(DevicesPointer devicesPointer, int index) {
        Pointer device = devicesPointer.getDevicePointer(index);

        int workItemDims = OpenCLHelper.getDeviceInfoInt(device, DeviceInfoCode.CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
        SizeT[] workItemSizes = new SizeT[workItemDims];
        Pointer sizesP = OpenCLHelper.getDeviceInfoPointer(device, DeviceInfoCode.CL_DEVICE_MAX_WORK_ITEM_SIZES);
        for (int i = 0; i < workItemSizes.length; i++) {
            workItemSizes[i] = SizeT.getSizeT(sizesP, i);
        }

        return new Device.Info(
                OpenCLHelper.getDeviceInfoString(device, DeviceInfoCode.CL_DEVICE_NAME),
                OpenCLHelper.getDeviceInfoInt(device, DeviceInfoCode.CL_DEVICE_MAX_COMPUTE_UNITS),
                OpenCLHelper.getDeviceInfoLong(device, DeviceInfoCode.CL_DEVICE_DOUBLE_FP_CONFIG),
                OpenCLHelper.getDeviceInfoLong(device, DeviceInfoCode.CL_DEVICE_GLOBAL_MEM_SIZE),
                OpenCLHelper.getDeviceInfoSizeT(device, DeviceInfoCode.CL_DEVICE_MAX_WORK_GROUP_SIZE),
                workItemSizes,
                OpenCLHelper.getDeviceInfoType(device, DeviceInfoCode.CL_DEVICE_TYPE));
    }
}
