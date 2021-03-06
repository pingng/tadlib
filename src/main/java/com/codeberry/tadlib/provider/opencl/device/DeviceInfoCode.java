package com.codeberry.tadlib.provider.opencl.device;

public enum DeviceInfoCode {
    /* cl_device_info */
    CL_DEVICE_TYPE(0x1000),
    CL_DEVICE_VENDOR_ID(0x1001),
    CL_DEVICE_MAX_COMPUTE_UNITS(0x1002),
    CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS(0x1003),
    CL_DEVICE_MAX_WORK_GROUP_SIZE(0x1004),
    CL_DEVICE_MAX_WORK_ITEM_SIZES(0x1005),
    CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR(0x1006),
    CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT(0x1007),
    CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT(0x1008),
    CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG(0x1009),
    CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT(0x100A),
    CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE(0x100B),
    CL_DEVICE_MAX_CLOCK_FREQUENCY(0x100C),
    CL_DEVICE_ADDRESS_BITS(0x100D),
    CL_DEVICE_MAX_READ_IMAGE_ARGS(0x100E),
    CL_DEVICE_MAX_WRITE_IMAGE_ARGS(0x100F),
    CL_DEVICE_MAX_MEM_ALLOC_SIZE(0x1010),
    CL_DEVICE_IMAGE2D_MAX_WIDTH(0x1011),
    CL_DEVICE_IMAGE2D_MAX_HEIGHT(0x1012),
    CL_DEVICE_IMAGE3D_MAX_WIDTH(0x1013),
    CL_DEVICE_IMAGE3D_MAX_HEIGHT(0x1014),
    CL_DEVICE_IMAGE3D_MAX_DEPTH(0x1015),
    CL_DEVICE_IMAGE_SUPPORT(0x1016),
    CL_DEVICE_MAX_PARAMETER_SIZE(0x1017),
    CL_DEVICE_MAX_SAMPLERS(0x1018),
    CL_DEVICE_MEM_BASE_ADDR_ALIGN(0x1019),
    CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE(0x101A),
    CL_DEVICE_SINGLE_FP_CONFIG(0x101B),
    CL_DEVICE_GLOBAL_MEM_CACHE_TYPE(0x101C),
    CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE(0x101D),
    CL_DEVICE_GLOBAL_MEM_CACHE_SIZE(0x101E),
    CL_DEVICE_GLOBAL_MEM_SIZE(0x101F),
    CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE(0x1020),
    CL_DEVICE_MAX_CONSTANT_ARGS(0x1021),
    CL_DEVICE_LOCAL_MEM_TYPE(0x1022),
    CL_DEVICE_LOCAL_MEM_SIZE(0x1023),
    CL_DEVICE_ERROR_CORRECTION_SUPPORT(0x1024),
    CL_DEVICE_PROFILING_TIMER_RESOLUTION(0x1025),
    CL_DEVICE_ENDIAN_LITTLE(0x1026),
    CL_DEVICE_AVAILABLE(0x1027),
    CL_DEVICE_COMPILER_AVAILABLE(0x1028),
    CL_DEVICE_EXECUTION_CAPABILITIES(0x1029),
    CL_DEVICE_QUEUE_PROPERTIES(0x102A),
    CL_DEVICE_NAME(0x102B),
    CL_DEVICE_VENDOR(0x102C),
    CL_DRIVER_VERSION(0x102D),
    CL_DEVICE_PROFILE(0x102E),
    CL_DEVICE_VERSION(0x102F),
    CL_DEVICE_EXTENSIONS(0x1030),
    CL_DEVICE_PLATFORM(0x1031),
    CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF(0x1034),
    CL_DEVICE_HOST_UNIFIED_MEMORY(0x1035),
    CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR(0x1036),
    CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT(0x1037),
    CL_DEVICE_NATIVE_VECTOR_WIDTH_INT(0x1038),
    CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG(0x1039),
    CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT(0x103A),
    CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE(0x103B),
    CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF(0x103C),
    CL_DEVICE_OPENCL_C_VERSION(0x103D),
    /* cl_khr_fp64 extension - no extension #define since it has no functions  */
    CL_DEVICE_DOUBLE_FP_CONFIG(0x1032),
    /* cl_khr_fp16 extension - no extension #define since it has no functions  */
    CL_DEVICE_HALF_FP_CONFIG(0x1033);

    public final int code;

    DeviceInfoCode(int code) {
        this.code = code;
    }
}