package com.codeberry.tadlib.provider.opencl.kernel;

/* cl_kernel_work_group_info */
public enum KernelWorkGroupInfoCode {
    // size_t
    CL_KERNEL_WORK_GROUP_SIZE(0x11B0),
    // size_t[3]
    CL_KERNEL_COMPILE_WORK_GROUP_SIZE(0x11B1),
    // cl_ulong
    CL_KERNEL_LOCAL_MEM_SIZE(0x11B2),
    // size_t
    CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE(0x11B3),
    // cl_ulong
    CL_KERNEL_PRIVATE_MEM_SIZE(0x11B4),
    // size_t[3]
    CL_KERNEL_GLOBAL_WORK_SIZE(0x11B5);

    public final int code;

    KernelWorkGroupInfoCode(int code) {
        this.code = code;
    }
}
