long calcOffset(long offset, int dimCount,  __global const int *toIndexBCS, __global const int *toOffsetBCS) {
    long r = 0;
    for(int i=0; i<dimCount; i++) {
        int toIndexSize = toIndexBCS[i];

        if(toIndexSize > 0) {
            int dimIndex = offset / toIndexSize;

            r += dimIndex * toOffsetBCS[i];

            offset %= toIndexSize;
        }
    }
    return r;
}

__kernel void tensorSum(int dimCount,
                        __global const double *src,
                        long srcSize,
                        long sumSize,
                        __global const int *inBCS,
                        __global const int *sumBCS,
                        __global const int *outBCS,
                        int aggregateValsPerWorker,
                        __global double *out,
                        __local double *working) {
    long localId = get_local_id(0);
    size_t i = get_global_id(0);
    long outIndex = get_global_id(1);
    long groupSize = get_local_size(0);

    if(i < sumSize) {

        long fixedOffset = calcOffset((long) get_global_id(1), dimCount, outBCS, inBCS);

        double sum = 0;
        for(int aI=0; aI < aggregateValsPerWorker; aI++) {
            long rawSumOffset = localId + aI * groupSize;

            if(rawSumOffset < sumSize) {

                long srcValOffset = calcOffset(rawSumOffset, dimCount, sumBCS, inBCS);
                long srcOffset = fixedOffset + srcValOffset;
                double val = src[srcOffset];
//                printf("%lld: localId=%lld rawSumOffset=%d: %f\n", outIndex, localId, rawSumOffset, val);
                sum += val;
            }
        }
        working[localId] = sum;
    } else {
        working[localId] = 0.0;
//        printf("%d: IGNORED\n", localId);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // --- SUMMATION ---
    if(groupSize > 1) {
        int limit = groupSize;
        do {
            // For _1st_ worker...
            if(localId == 0 && (limit & 1) != 0) {
                //...when groupSize is an odd number, let the first worker add the last item
                working[0] += working[limit-1];
            }
            limit >>= 1;
            if(localId < limit) {
                working[localId] += working[localId + limit];

//                if(localId == 0) {
//                    printf("Id: %d, %f\n", limit, working[localId]);
//                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        } while(limit > 1);
    }
    if(localId == 0) {
        out[outIndex] = working[0];
    }

}
