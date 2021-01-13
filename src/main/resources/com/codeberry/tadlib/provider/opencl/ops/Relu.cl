__kernel void relu(__global const double *src,
                    double leakyScale,
                    __global double *out,
                    __global double *outMask,
                    long outLen) {
    long offset = get_global_id(0);

    if(offset < outLen) {
        double v = src[offset];
        if(v >= 0) {
            out[offset] = v;
            outMask[offset] = 1.0;
        } else {
            out[offset] = v * leakyScale;
            outMask[offset] = leakyScale;
        }
    }
}
