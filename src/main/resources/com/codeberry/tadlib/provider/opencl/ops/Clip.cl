__kernel void clip(__global const double *src,
                    long srcLen,
                    int hasMin,
                    int hasMax,
                    double minV,
                    double maxV,
                    __global double *out
                    ) {
    long offset = get_global_id(0);

    if(offset < srcLen) {
        double v = src[offset];

        if(hasMin)
            v = max(minV, v);
        if(hasMax)
            v = min(maxV, v);

        out[offset] = v;
    }
}
