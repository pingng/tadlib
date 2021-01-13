__kernel void softmax(__global const double *src,
                    int valuesPerExample,
                    __global double *out,
                    __local double *vals) {
    long unitIndex = get_local_id(0);

    long exampleIndex = get_global_id(1);
    long offset = exampleIndex * valuesPerExample + unitIndex;

    double v = -DBL_MAX;
    if(unitIndex < valuesPerExample) {
        v = src[offset];
    }
    vals[unitIndex] = v;
//    if(unitIndex < valuesPerExample) printf("Val: %llf\n", v);

    // --- Find max ---
    barrier(CLK_LOCAL_MEM_FENCE);

    if(valuesPerExample > 1) {
        int limit = valuesPerExample;
        do {
            // For _1st_ worker...
            if(unitIndex == 0 && (limit & 1) != 0) {
                //...when groupSize is an odd number, let the first worker add the last item
                vals[0] = max(vals[0], vals[limit-1]);
            }
            limit >>= 1;
            if(unitIndex < limit) {
                vals[unitIndex] = max(vals[unitIndex], vals[unitIndex + limit]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        } while(limit > 1);
    }
    double max = vals[0];
//    if(unitIndex < valuesPerExample) printf("Max: %llf\n", max);

    // --- Calc shifted exp ---
    barrier(CLK_LOCAL_MEM_FENCE);

    double exped = 0;
    if(unitIndex < valuesPerExample) {
        double shifted = v - max;
        exped = exp(shifted);
//        printf("Exped[%lld]: %llf\n", unitIndex, exped);
    }
    vals[unitIndex] = exped;

    // --- Sum Exp ---
    barrier(CLK_LOCAL_MEM_FENCE);

    if(valuesPerExample > 1) {
        int limit = valuesPerExample;
        do {
            // For _1st_ worker...
            if(unitIndex == 0 && (limit & 1) != 0) {
                //...when groupSize is an odd number, let the first worker add the last item
//                printf("SUM: [%lld & %d]: %llf + %llf = %llf\n", unitIndex, limit-1, vals[0], vals[limit-1], vals[0]+ vals[limit-1]);
                vals[0] += vals[limit-1];
            }
            limit >>= 1;
            if(unitIndex < limit) {
//                printf("Sum: [%lld & %d]: %llf + %llf = %llf\n", unitIndex, unitIndex+limit, vals[unitIndex], vals[unitIndex + limit], vals[unitIndex]+ vals[unitIndex + limit]);
                vals[unitIndex] += vals[unitIndex + limit];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        } while(limit > 1);
    }

    // --- Output ---
    if(unitIndex < valuesPerExample) {
        double expSum = vals[0];
//        printf("expSum: %llf\n", expSum);

        out[offset] = exped / expSum;
//        printf("softmax: %llf\n", out[offset]);
    }


}
