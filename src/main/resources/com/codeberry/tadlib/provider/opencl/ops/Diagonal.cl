__kernel void diag(__global const double *src,
                    int srcDimCount,
                    __global int *srcBlockSizes,
                    long srcSize,
                    __global double *out,
                    __global int *outBroadcastBlockSizes) {
    long srcOffset = get_global_id(0);

    if(srcOffset < srcSize) {
        long outOffset = 0;
        long currentOffset = srcOffset;
        long index = 0;
        for(int i=0; i<srcDimCount; i++) {
            int blockSize = srcBlockSizes[i];
            index = currentOffset / blockSize;

            outOffset += index * outBroadcastBlockSizes[i];

            currentOffset %= blockSize;
        }
        // Add the last out index, which is the same as the last src index (it's the diagonal)
        outOffset += index * outBroadcastBlockSizes[srcDimCount];

        out[outOffset] = src[srcOffset];
    }
}
