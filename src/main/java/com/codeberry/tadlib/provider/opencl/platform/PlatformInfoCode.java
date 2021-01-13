package com.codeberry.tadlib.provider.opencl.platform;

import java.util.Collections;
import java.util.Map;
import java.util.stream.Collectors;

import static java.util.Arrays.stream;
import static java.util.function.Function.identity;

public enum PlatformInfoCode {
    /* cl_platform_info */
    CL_PLATFORM_PROFILE(0x0900),
    CL_PLATFORM_VERSION(0x0901),
    CL_PLATFORM_NAME(0x0902),
    CL_PLATFORM_VENDOR(0x0903),
    CL_PLATFORM_EXTENSIONS(0x0904);

    public final int code;

    PlatformInfoCode(int code) {
        this.code = code;
    }
}
