__kernel void updateDouble(__global double *out,
                            long outLen,
                            __global const long *offsets,
                            __global const double *values,
                            int updateCount) {
    long updateId = get_global_id(0);

    if(updateId < updateCount) {
        double v = values[updateId];
        long outOffset = offsets[updateId];

        if(outOffset >= outLen) printf("OUT OF BOUNDS!!!\n");

        out[outOffset] = v;
    }
}
