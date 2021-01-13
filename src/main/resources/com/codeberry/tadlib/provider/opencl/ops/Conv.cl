#define FLI_W_INCHAN            0
#define FLI_INCHAN              1
#define OUT_BS_W_OUTCHAN        0
#define OUT_BS_OUTCHAN          1
#define IN_SIZE_HEIGHT           0
#define IN_SIZE_WIDTH            1
#define IN_SIZE_CHANS            2
#define OFFSET_Y                0
#define OFFSET_X                1


__kernel void conv2d(__global const double *inputs,
                     long inputLength,
                     __global const int *inputSizes,
                     __global const double *filter,
                     long filterLength,
                     __global const int *filterOffsets,
                     int singleOutputFilterVolume,
                     int inputExampleBlockSize,
                     __global const int *outputBlockSize,
                     __global const int *filterBlockSize,
                     __global double *out,
                     int outputExampleBlockSize,
                     int filterCalcPerWorkItem,
                     int workingLength,
                     __local double *working) {
    const int inputHeight = inputSizes[IN_SIZE_HEIGHT];
    const int inputWidth = inputSizes[IN_SIZE_WIDTH];
    const int inputChannelCount = inputSizes[IN_SIZE_CHANS];
    const int outputChannelCount = outputBlockSize[OUT_BS_OUTCHAN];

    size_t tmp;
    size_t rawExampleIndex = get_global_id(2);
    unsigned long exampleOffset = rawExampleIndex * inputExampleBlockSize;
    unsigned long outputExampleOffset = rawExampleIndex * outputExampleBlockSize;

    size_t rawPositionIndex = get_global_id(1);
    tmp = rawPositionIndex;
    unsigned int outY = tmp / outputBlockSize[OUT_BS_W_OUTCHAN];
    tmp %= outputBlockSize[OUT_BS_W_OUTCHAN];
    unsigned int outX = tmp / outputBlockSize[OUT_BS_OUTCHAN];
    tmp %= outputBlockSize[OUT_BS_OUTCHAN];
    unsigned int outChannel = tmp;

    long finalOutIndex = outputExampleOffset +
            outY * outputBlockSize[OUT_BS_W_OUTCHAN] +
            outX * outputBlockSize[OUT_BS_OUTCHAN] +
            outChannel;

    size_t localId = get_local_id(0);

    double sum = 0;
    size_t rawFilterIndex = get_global_id(0) * filterCalcPerWorkItem;
    for(int filterCalcIdx=0; filterCalcIdx<filterCalcPerWorkItem; filterCalcIdx++) {
        if(filterCalcIdx < singleOutputFilterVolume) {
            tmp = rawFilterIndex + filterCalcIdx;
            unsigned int filterY = tmp / filterBlockSize[FLI_W_INCHAN];
            tmp %= filterBlockSize[FLI_W_INCHAN];
            unsigned int filterX = tmp / filterBlockSize[FLI_INCHAN];
            tmp %= filterBlockSize[FLI_INCHAN];
            unsigned int inChannel = tmp;

            unsigned int inY = filterY + filterOffsets[OFFSET_Y] + outY;
            unsigned int inX = filterX + filterOffsets[OFFSET_X] + outX;

            if(inY >= 0 && inY < inputHeight &&
                inX >= 0 && inX < inputWidth) {
                long _fIndex = filterY * filterBlockSize[FLI_W_INCHAN] * outputChannelCount +
                               filterX * filterBlockSize[FLI_INCHAN] * outputChannelCount +
                               inChannel * outputChannelCount +
                               outChannel;
                if(_fIndex < filterLength) {
                    long _inputIdx = exampleOffset +
                                         inY * inputWidth * inputChannelCount +
                                         inX * inputChannelCount +
                                         inChannel;
                    double inValue = inputs[_inputIdx];
                    double filterValue = filter[_fIndex];
                    double _prod = inValue * filterValue;

                    if(isnan(_prod)) {
                        printf("CONV2D NAN: (%llf) %lld of %lld, filter (%llf) %lld of %lld\n",
                            inValue, _inputIdx, inputLength, filterValue, _fIndex, filterLength);
                    }
                    sum += _prod;

//                    if(rawExampleIndex == 0 && outX == 4 && outY == 0 && isnan(filterValue)) {
//                        printf("***NAN filter: %llu: (%u, %u)\n", rawFilterIndex, filterX, filterY);
    //                        printf("ExampleOffset %llu:\n"
    //                                 "  Output %llu: (%u, %u, %u)\n"
    //                                 "  Input: (%u, %u)\n"
    //                                 "  Filter %llu: (%u, %u, %u)\n"
    //                                 "  Values: %f x %f = %f\n"
    //                                 "  Local %llu\n",
    //                                 rawExampleIndex,
    //                            rawPositionIndex, outX, outY, outChannel,
    //                            inX, inY,
    //                            rawFilterIndex,filterX, filterY, inChannel,
    //                            inValue, filterValue, inValue*filterValue,
    //                            localId);
//                    }
                }
            } else {
//                if(outX == 0 && outY == 0) {
//                    printf("* IGNORED ExampleOffset %llu:\n"
//                             "  Output %llu: (%u, %u, %u)\n"
//                             "  Input: (%d, %d)\n"
//                             "  Filter %llu: (%u, %u, %u)\n"
//                             "  Local %llu\n",
//                             rawExampleIndex,
//                        rawPositionIndex, outX, outY, outChannel,
//                        inX, inY,
//                        rawFilterIndex,filterX, filterY, inChannel,
//                        localId);
//                }
            }
        }
    }
    working[localId] = sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    // --- SUMMATION ---
    if(workingLength > 1) {
        int limit = workingLength;
        do {
            // For _1st_ worker...
            if(localId == 0 && (limit & 1) != 0) {
                //...when workingLength is an odd number, let the first worker add the last item
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
        out[finalOutIndex] = working[0];
    }

}