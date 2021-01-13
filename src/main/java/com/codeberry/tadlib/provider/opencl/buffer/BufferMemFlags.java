package com.codeberry.tadlib.provider.opencl.buffer;

public enum BufferMemFlags {
    /* cl_mem_flags - bitfield */
    CL_MEM_READ_WRITE(1 << 0),
    CL_MEM_WRITE_ONLY(1 << 1),
    CL_MEM_READ_ONLY(1 << 2),
    CL_MEM_USE_HOST_PTR(1 << 3),
    CL_MEM_ALLOC_HOST_PTR(1 << 4),
    CL_MEM_COPY_HOST_PTR(1 << 5);

    public final long bits;

    BufferMemFlags(long bits) {
        this.bits = bits;
    }
}
