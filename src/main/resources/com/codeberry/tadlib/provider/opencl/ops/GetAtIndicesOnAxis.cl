__kernel void getIndicesOnAxis(__global const double *src,
                          long srcLen,
                          __global const int *srcBroadcastBlockSizes,
                          int axis,
                          __global const int *indices,
                          __global double *out,
                          long outLen,
                          int outDimCount,
                          __global const int *outBlockSizes) {

    long outOffset = get_global_id(0);

    if(outOffset < outLen) {
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
        long finalSrcOffset = srcOffset +
                                indices[outOffset] * srcBroadcastBlockSizes[axis];
        //printf("outOffset: %lld, %d, axis: %d, %d, final=%lld, val=%llf\n",
        //    outOffset, indices[outOffset], axis, srcBroadcastBlockSizes[axis], finalSrcOffset,
        //    src[finalSrcOffset]);
        // ---

        out[outOffset] = src[finalSrcOffset];
    }
}
