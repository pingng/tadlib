__kernel void argMax(__global const double *src,
                          long srcLen,
                          __global const int *srcBroadcastBlockSizes,
                          int axis,
                          int axisLen,
                          int valuesToCheckPerWorker,
                          __global int *out,
                          long outLen,
                          int outDimCount,
                          __global const int *outBlockSizes,
                          __local int *workingIndices,
                          __local double *workingValues) {
    long workerId = get_local_id(0);
    long workerCount = get_local_size(0);
    long outOffset = get_global_id(1);

    // ---
    long srcOffset = 0;
    long currentOutOffset = outOffset;
    for(int j=0; j<outDimCount; j++) {
        int outBlockSize = outBlockSizes[j];
        long index = currentOutOffset / outBlockSize;

        int srcJ = (j < axis ? j : j + 1);
        srcOffset += index * srcBroadcastBlockSizes[srcJ];

        currentOutOffset %= outBlockSize;
    }
    // ---


    int myMaxIndex = INT_MIN;
    double myMaxValue = -DBL_MAX;

    for(int i=0; i<valuesToCheckPerWorker; i++) {
        long myAxisIndex = workerId + i * workerCount;

        if(myAxisIndex < axisLen) {
            long finalSrcOffset = srcOffset +
                                    myAxisIndex * srcBroadcastBlockSizes[axis];
            double srcVal = src[finalSrcOffset];

            if(srcVal > myMaxValue) {
                myMaxValue = srcVal;
                myMaxIndex = myAxisIndex;
            }
        }
    }

    workingIndices[workerId] = myMaxIndex;
    workingValues[workerId] = myMaxValue;

    // --- Find max ---
    barrier(CLK_LOCAL_MEM_FENCE);

    int limit = workerCount;
    do {
        // For _1st_ worker...
        if(workerId == 0 && (limit & 1) != 0) {
            //...when groupSize is an odd number, let the first worker check the last item
            if(workingValues[limit-1] > workingValues[0]) {
                workingValues[0] = workingValues[limit-1];
                workingIndices[0] = workingIndices[limit-1];
            }
        }
        limit >>= 1;
        if(workerId < limit) {
            if(workingValues[workerId + limit] > workingValues[workerId]) {
                workingValues[workerId] = workingValues[workerId + limit];
                workingIndices[workerId] = workingIndices[workerId + limit];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    } while(limit > 1);

    if(workerId == 0) {
        out[outOffset] = workingIndices[0];
    }
}
