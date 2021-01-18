__kernel void updateAtIndicesOnAxis(
                          long outLen,
                          __global const int *outBroadcastBlockSizes,
                          int axis,
                          int axisLen,
                          __global const int *indices,
                          __global const double *change,
                          __global double *out,
                          long indicesLen,
                          int indicesDimCount,
                          __global const int *indicesBlockSizes) {
    long indicesOffset = get_global_id(0);

    if(indicesOffset < indicesLen) {
        // ---
        long outOffset = 0;
        long currentIndicesOffset = indicesOffset;
        for(int j=0; j<indicesDimCount; j++) {
            int indBlockSize = indicesBlockSizes[j];
            long index = currentIndicesOffset / indBlockSize;
            int outJ = (j < axis ? j : j + 1);
            outOffset += index * outBroadcastBlockSizes[outJ];

            currentIndicesOffset %= indBlockSize;

        }
        int axisIndex = indices[indicesOffset];
        if(axisIndex >= 0 && axisIndex < axisLen) {
            long finalOutOffset = outOffset + axisIndex * outBroadcastBlockSizes[axis];
    //        printf("outOffset: %lld, %d, axis: %d, %d, final=%lld, val=%llf\n",
    //            outOffset, indices[indicesOffset], axis, outBroadcastBlockSizes[axis], finalOutOffset,
    //            change[indicesOffset]);
            // ---

            out[finalOutOffset] = change[indicesOffset];
        } else {
            printf("axisIndex is out of bounds: indexValue=%d indexOffset=%lld\n", axisIndex, indicesOffset);
            // Just put NAN in an element so we can detect the error
            long finalOutOffset = outOffset;
            out[finalOutOffset] = NAN;
        }
    }
}
