__kernel void transpose(__global const double *src,
                     __global const int *srcBlockSizes,
                     __global double *out,
                     __global const int *outBroadcastBlockSizes,
                     int outDimCount,
                     long outLen,
                     int workGroupWidth,
                     __local double *working) {

    long srcRawIndex = get_global_id(0);

    if(srcRawIndex < outLen) {
        long currentRaw = srcRawIndex;

        int offsetOut = 0;
        for(int dimI=0; dimI<outDimCount; dimI++) {
            int srcBlockSize = srcBlockSizes[dimI];
            int srcIndex = currentRaw / srcBlockSize;

            int outBroadcastBlockSize = outBroadcastBlockSizes[dimI];
            offsetOut += srcIndex * outBroadcastBlockSize;

//            printf("%lld: srcBlockSize=%d, outBlockSize=%d, currentRaw=%d srcIndex=%d\n",
//                    srcRawIndex, srcBlockSize, outBroadcastBlockSize, currentRaw, srcIndex);
            currentRaw %= srcBlockSize;
        }
        double val = src[srcRawIndex];

        if(isnan(val)) {
            printf("Transpose NAN! index: %lld\n", srcRawIndex);
        }

        out[offsetOut] = val;

//        printf("out[%d] = src[%lld]\n", offsetOut, srcRawIndex);
    }

}