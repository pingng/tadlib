package com.codeberry.tadlib.provider.opencl.platform;

import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Pointer;

import static com.codeberry.tadlib.provider.opencl.OpenCLHelper.getPlatformInfoString;
import static com.codeberry.tadlib.provider.opencl.platform.PlatformInfoCode.CL_PLATFORM_NAME;
import static com.codeberry.tadlib.provider.opencl.platform.PlatformInfoCode.CL_PLATFORM_VERSION;

public class PlatformsPointer extends Memory {
    private final int platformCount;

    public PlatformsPointer(int platformCount) {
        super((long) Native.POINTER_SIZE * platformCount);
        this.platformCount = platformCount;
    }

    int getPlatformCount() {
        return platformCount;
    }

    Pointer getPlatFormPointer(int index) {
        return getPointer((long) index * Native.POINTER_SIZE);
    }

    Platform.Info getPlatformInfo(PlatformsPointer platformsPointer, int index) {
        Pointer platform = platformsPointer.getPlatFormPointer(index);

        return new Platform.Info(
                getPlatformInfoString(platform, CL_PLATFORM_NAME),
                getPlatformInfoString(platform, CL_PLATFORM_VERSION));
    }
}
