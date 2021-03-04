__global double* getOut(int index,
                        __global double *out_0,
                        __global double *out_1,
                        __global double *out_2,
                        __global double *out_3,
                        __global double *out_4,
                        __global double *out_5,
                        __global double *out_6,
                        __global double *out_7,
                        __global double *out_8,
                        __global double *out_9) {
    switch(index) {
    case 0: return out_0;
    case 1: return out_1;
    case 2: return out_2;
    case 3: return out_3;
    case 4: return out_4;
    case 5: return out_5;
    case 6: return out_6;
    case 7: return out_7;
    case 8: return out_8;
    case 9: return out_9;
    }
    printf("OUT INDEX OUT OF BOUNDS: %d\n", index);
}

__kernel void split(__global const double *src,
                      int srcDimCount,
                      __global const int *srcDimArray,
                      long srcSize,
                      int axis,
                      __global const int *axisLens,
                      __global double *out_0,
                      __global double *out_1,
                      __global double *out_2,
                      __global double *out_3,
                      __global double *out_4,
                      __global double *out_5,
                      __global double *out_6,
                      __global double *out_7,
                      __global double *out_8,
                      __global double *out_9) {
    long srcOffset = get_global_id(0);

    long curOutBlockSize = 1;
    long outOffset = 0;
    int myOutIndex = -1;

    if(srcOffset < srcSize) {
        // Iterate last->first axis
        long currentSrcOffset = srcOffset;
        for(int i=srcDimCount-1; i>=0; i--) {
            int currentSrcAxisLen = srcDimArray[i];
            int index = currentSrcOffset % currentSrcAxisLen;

            int currentOutAxisLen = currentSrcAxisLen;
            if(i == axis) {
                myOutIndex = 0;
                while(index >= axisLens[myOutIndex]) {
                    index -= axisLens[myOutIndex];
                    myOutIndex++;
                }
                currentOutAxisLen = axisLens[myOutIndex];
            }

            outOffset += index * curOutBlockSize;

            curOutBlockSize *= currentOutAxisLen;
            currentSrcOffset /= currentSrcAxisLen;
        }

        __global double* out = getOut(myOutIndex, out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9);

        out[outOffset] = src[srcOffset];
    }
}
