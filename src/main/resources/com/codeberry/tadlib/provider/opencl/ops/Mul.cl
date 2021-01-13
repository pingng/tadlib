__kernel void tensorMulScalar(__global const double *A, double scalar, long len,
                              __global double *C) {
    size_t i = get_global_id(0);

    // Do the operation
    if( i < len ) {
        C[i] = A[i] * scalar;
    }
}

__kernel void singleElementMul(__global const double *A, __global const double *B, __global double *C) {
    C[0] = A[0] * B[0];
}

__kernel void arrayMul(__global const double *A, __global const double *B, long len,
                        __global double *C) {
    size_t i = get_global_id(0);

    // Do the operation
    if( i < len ) {
        C[i] = A[i] * B[i];
    }
}

__kernel void tensorMul(__global const double *A,
                        __global const double *B,
                        __global const int *dimBlockSizesA,
                        __global const int *dimBlockSizesB,
                        __global const int *dimBlockSizesOut,
                        int outDimCount,
                        __global double *out,
                        long outLen) {
    size_t outRawIndex = get_global_id(0);
    if(outRawIndex < outLen) {
        size_t currentRaw = outRawIndex;

        int indexA = 0;
        int indexB = 0;
        for(int dimI=0; dimI<outDimCount; dimI++) {
            int blockSizeOut = dimBlockSizesOut[dimI];
            int dimIndexOut = currentRaw / blockSizeOut;

            indexA += dimIndexOut * dimBlockSizesA[dimI];
            indexB += dimIndexOut * dimBlockSizesB[dimI];

            currentRaw %= blockSizeOut;
        }

        out[outRawIndex] = A[indexA] * B[indexB];

        //printf("out dim %d, size  %lld, %ld\n", outDimCount, outLen, outRawIndex);
    }
}