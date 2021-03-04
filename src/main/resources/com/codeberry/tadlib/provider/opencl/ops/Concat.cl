__global const double* getSrc(int index,
                                __global const double *src_0,
                                __global const double *src_1,
                                __global const double *src_2,
                                __global const double *src_3,
                                __global const double *src_4,
                                __global const double *src_5,
                                __global const double *src_6,
                                __global const double *src_7,
                                __global const double *src_8,
                                __global const double *src_9) {
    switch(index) {
    case 0: return src_0;
    case 1: return src_1;
    case 2: return src_2;
    case 3: return src_3;
    case 4: return src_4;
    case 5: return src_5;
    case 6: return src_6;
    case 7: return src_7;
    case 8: return src_8;
    case 9: return src_9;
    }
    printf("SOURCE INDEX OUT OF BOUNDS: %d\n", index);
}


__kernel void concat(__global const double *src_0,
                    __global const double *src_1,
                    __global const double *src_2,
                    __global const double *src_3,
                    __global const double *src_4,
                    __global const double *src_5,
                    __global const double *src_6,
                    __global const double *src_7,
                    __global const double *src_8,
                    __global const double *src_9,
                      int srcArrCount,
                      int axis,
                      __global const int *axisLens,
                      __global double *out,
                      int outDimCount,
                      __global const int *outDimArray,
                      long outSize) {
    long outOffset = get_global_id(0);

    long curSrcBlockSize = 1;
    long srcOffset = 0;
    int mySrcIndex = -1;

    if(outOffset < outSize) {
        // Iterate last->first axis
        long currentOutOffset = outOffset;
        for(int i=outDimCount-1; i>=0; i--) {
            int currentOutAxisLen = outDimArray[i];
            int index = currentOutOffset % currentOutAxisLen;

            int currentSrcAxisLen = currentOutAxisLen;
            if(i == axis) {
                mySrcIndex = 0;
                while(index >= axisLens[mySrcIndex]) {
                    index -= axisLens[mySrcIndex];
                    mySrcIndex++;
                }
                currentSrcAxisLen = axisLens[mySrcIndex];
            }

            srcOffset += index * curSrcBlockSize;

            curSrcBlockSize *= currentSrcAxisLen;
            currentOutOffset /= currentOutAxisLen;
        }

        __global const double* src = getSrc(mySrcIndex, src_0, src_1, src_2, src_3, src_4, src_5, src_6, src_7, src_8, src_9);

        out[outOffset] = src[srcOffset];
    }
}
