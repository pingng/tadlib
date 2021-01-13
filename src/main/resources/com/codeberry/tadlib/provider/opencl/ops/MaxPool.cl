__kernel void maxPool2d(int maxPoolSize,
                        int channels,
                        __global const double *src,
                        int src2dArea,
                        int src2dWidth,
                        int src2dHeight,
                        int inputsPerOutputValue,
                        int out2dArea,
                        int out2dWidth,
                        __global double *out,
                        __global int *outId,
                        __local double *working,
                        __local int *workingId) {
    long rawOutOffset = get_global_id(1);
    int localId = get_local_id(0);
    long groupSize = get_local_size(0);

    workingId[localId] = localId;
    double value = -DBL_MAX;
    if(localId < inputsPerOutputValue) {
        int outChannel = rawOutOffset % channels;
        long rawOutAreaOffset = rawOutOffset / channels;
        long out2dAreaOffset = rawOutAreaOffset % out2dArea;
        int out2dY = out2dAreaOffset / out2dWidth;
        int out2dX = out2dAreaOffset % out2dWidth;

        long out2dPlaneOffset = rawOutAreaOffset / out2dArea;
        long src2dPlaneOffset = out2dPlaneOffset * src2dArea;
        int src2dY = (localId / maxPoolSize) + out2dY * maxPoolSize;
        int src2dX = (localId % maxPoolSize) + out2dX * maxPoolSize;

        if(src2dY < src2dHeight && src2dX < src2dWidth) {
            int localViewOffset = (src2dPlaneOffset +
                                    src2dY * src2dWidth+
                                    src2dX) * channels +
                                    outChannel;
            double srcValue = src[localViewOffset];
//            if(out2dX==1 && out2dY==1) {
//                printf("In  %lld: %lld, (%d, %d) - %d = %llf\n"
//                       "Out  : %lld, (%d, %d) - %d\n",
//                        localId, out2dPlaneOffset, src2dX, src2dY,  outChannel, srcValue,
//                        out2dPlaneOffset, out2dX, out2dY, outChannel);
//            }
            value = srcValue;
            if(isnan(value)) {
                printf("NAN! %d\n",localViewOffset);
            }
        }
    }
    working[localId] = value;

    barrier(CLK_LOCAL_MEM_FENCE);

    // --- MAX AMONG THE GROUP ---
    if(groupSize > 1) {
        int limit = groupSize;
        do {
            // For _1st_ worker...
            if(localId == 0 && (limit & 1) != 0) {
                //...when groupSize is an odd number, let the first worker add the last item
                if(working[0] < working[limit-1]) {
                    working[0] = working[limit-1];
                    workingId[0] = workingId[limit-1];
                }
            }
            limit >>= 1;
            if(localId < limit) {
                if(working[localId] < working[localId + limit]) {
                    working[localId] = working[localId + limit];
                    workingId[localId] = workingId[localId + limit];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        } while(limit > 1);
    }
    if(localId == 0) {
        out[rawOutOffset] = working[0];
        outId[rawOutOffset] = workingId[0];
    }

}

__kernel void maxPool2dRevert(int maxPoolSize,
                        int channels,
                        __global const double *gradSrc,
                        long gradSrcLen,
                        int grad2dArea,
                        int grad2dWidth,
                        __global const int *orgPoolId,
                        int org2dArea,
                        int org2dWidth,
                        __global double *orgOut) {
    long rawGradOffset = get_global_id(0);

    if(rawGradOffset < gradSrcLen) {
        long gradCoordOffset = rawGradOffset / channels;
        long outChannel = rawGradOffset % channels;
        long grad2dPlaneId = gradCoordOffset / grad2dArea;
        int grad2dOffset = gradCoordOffset % grad2dArea;
        int gradY = grad2dOffset / grad2dWidth;
        int gradX = grad2dOffset % grad2dWidth;

        long poolWindowsOffset = orgPoolId[rawGradOffset];
        int orgY = poolWindowsOffset / maxPoolSize;
        int orgX = poolWindowsOffset % maxPoolSize;

        long orgCoordOffset = grad2dPlaneId * org2dArea +
                                gradY * maxPoolSize * org2dWidth +
                                gradX * maxPoolSize +
                                orgY * org2dWidth +
                                orgX;
        long orgOffset = orgCoordOffset * channels +
                            outChannel;
        orgOut[orgOffset] = gradSrc[rawGradOffset];
    }
}