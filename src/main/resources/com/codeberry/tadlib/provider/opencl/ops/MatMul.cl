// NOTE: Each work group is organized to be optimized using local vars,
//       but the performance is good enough. This kernel should be ready
//       to be optimized if needed (workers are grouped in sensible groups).
__kernel void matmul(__global const double *left,
                     long leftLen,
                     __global const double *right,
                     long rightLen,
                     __global const int *leftBroadcastBlockSizes,
                     __global const int *rightBroadcastBlockSizes,
                     int mulsPerOutput,
                     int outDimCount,
                     __global const int *outDimSizes,
                     __global const int *outExampleBlockSizes,
                     __global double *out,
                     __global const int *outBroadcastBlockSizes
                     //__local double *leftBuf,
                     //__local double *rightBuf
                     ) {

    long outY = get_global_id(0);
    long outX = get_global_id(1);
    long exampleOffset = get_global_id(2);
    int outWidth = outDimSizes[outDimCount-1];
    int outHeight = outDimSizes[outDimCount-2];

    //printf("Size: (%d, %d), Pos: (%lld, %lld | %lld)\n", outWidth, outHeight, outX, outY, exampleOffset);

    if(outY < outHeight && outX < outWidth) {
        long leftExampleOffset = 0;
        long rightExampleOffset = 0;
        long outExampleOffset = 0;
        for(int i=0; i<outDimCount-2; i++) {
            long index = exampleOffset / outExampleBlockSizes[i];

            leftExampleOffset += index * leftBroadcastBlockSizes[i];
            rightExampleOffset += index * rightBroadcastBlockSizes[i];
            outExampleOffset += index * outBroadcastBlockSizes[i];

            exampleOffset %= outExampleBlockSizes[i];
        }

        double sum = 0;
        for(int i=0; i<mulsPerOutput; i++) {
            long leftOffset = leftExampleOffset +
                              outY * leftBroadcastBlockSizes[outDimCount-2] +
                              i    * leftBroadcastBlockSizes[outDimCount-1];
            long rightOffset = rightExampleOffset +
                               i    * rightBroadcastBlockSizes[outDimCount-2] +
                               outX * rightBroadcastBlockSizes[outDimCount-1];
            double leftValue = left[leftOffset];
            double rightValue = right[rightOffset];
//            printf("%f * %f\n", leftValue, rightValue);
            double m = leftValue * rightValue;

            sum += m;
        }

//        if(isnan(sum)) printf("Fin: %lld, %lld = %llf\n", outHeight, outY, sum);

        out[outExampleOffset +
                outY * outBroadcastBlockSizes[outDimCount-2] +
                outX * outBroadcastBlockSizes[outDimCount-1]] = sum;
    }

}