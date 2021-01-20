#define DT_DOUBLE 0
#define DT_INT 1
#define CMP_EQUALS 0
#define CMP_GREATER_EQUALS 1
#define CMP_LESS_EQUALS 2
#define CMP_GREATER 3
#define CMP_LESS 4

double getAsDouble(long offset, int dataType, __global const void *buffer) {
    if(dataType == DT_DOUBLE) {
        __global const double *dblBuf = (__global const double*) buffer;
        return dblBuf[offset];
    } else {
        //...then is int
        __global const int *intBuf = (__global const int*) buffer;
        return (double) intBuf[offset];
    }
}

int getAsInt(long offset, int dataType, __global const void *buffer) {
    if(dataType == DT_DOUBLE) {
        __global const double *dblBuf = (__global const double*) buffer;
        return (int) dblBuf[offset];
    } else {
        //...then is int
        __global const int *intBuf = (__global const int*) buffer;
        return intBuf[offset];
    }
}

bool compareDoubles(int comparisonCode, double leftValue, double rightValue, double comparisonDelta) {
    switch(comparisonCode) {
        case CMP_EQUALS: {
            double diff = rightValue - leftValue;
            if(diff < 0)
                diff = -diff;
            return diff <= comparisonDelta;
        }
        case CMP_GREATER_EQUALS:
            return leftValue >= rightValue;
        case CMP_LESS_EQUALS:
            return leftValue <= rightValue;
        case CMP_GREATER:
            return leftValue > rightValue;
        case CMP_LESS:
            return leftValue < rightValue;
    }

    printf("(double) Unknown comparison code: %d\n", comparisonCode);
    return false;
}

bool compareInts(int comparisonCode, int leftValue, int rightValue) {
    switch(comparisonCode) {
        case CMP_EQUALS:
            return leftValue == rightValue;
        case CMP_GREATER_EQUALS:
            return leftValue >= rightValue;
        case CMP_LESS_EQUALS:
            return leftValue <= rightValue;
        case CMP_GREATER:
            return leftValue > rightValue;
        case CMP_LESS:
            return leftValue < rightValue;
    }

    printf("(int) Unknown comparison code: %d\n", comparisonCode);
    return false;
}


__kernel void compare(__global const void *left,
                      int leftDataType,
                      __global const void *right,
                      int rightDataType,
                      __global const void *trueValue,
                      __global const void *falseValue,
                      __global const int *leftBroadcastBlockSizes,
                      __global const int *rightBroadcastBlockSizes,
                      int comparisonCode,
                      double comparisonDelta,
                      __global void *out,
                      int outDimCount,
                      __global const int *outBlockSizes,
                      long outLen
) {
    long outOffset = get_global_id(0);

    if(outOffset < outLen) {
        long leftOffset = 0;
        long rightOffset = 0;
        long currentOutOffset = outOffset;
        for(int i=0; i<outDimCount; i++) {
            int blockSize = outBlockSizes[i];
            int index = currentOutOffset / blockSize;

            leftOffset += index * leftBroadcastBlockSizes[i];
            rightOffset += index * rightBroadcastBlockSizes[i];

            currentOutOffset %= blockSize;
        }

        if(leftDataType == DT_DOUBLE || rightDataType == DT_DOUBLE) {
            double leftValue = getAsDouble(leftOffset, leftDataType, left);
            double rightValue = getAsDouble(rightOffset, rightDataType, right);
            double outValue = (compareDoubles(comparisonCode, leftValue, rightValue, comparisonDelta) ?
                                ((__global const double *)trueValue)[0] :
                                ((__global const double *)falseValue)[0]);

            __global double *dblOut = (__global double *)out;
            dblOut[outOffset] = outValue;
        } else {
            //...then left & right are ints, output is int
            int leftValue = getAsInt(leftOffset, leftDataType, left);
            int rightValue = getAsInt(rightOffset, rightDataType, right);
            int outValue = (compareInts(comparisonCode, leftValue, rightValue) ?
                                ((__global const int *)trueValue)[0] :
                                ((__global const int *)falseValue)[0]);

            __global int *intOut = (__global int *)out;
            intOut[outOffset] = outValue;
        }
    }

/*
    if(outOffset == 0) {
        if(dummyType == DT_DOUBLE) {
            __global const double *dummyDouble = (__global const double *)dummy;
            printf("Double: %llf, %llf\n", dummyDouble[0], dummyDouble[1]);
        } else if(dummyType == DT_INT) {
            __global const int *dummyInt = (__global const int *)dummy;
            printf("Int: %d, %d\n", dummyInt[0], dummyInt[1]);
        }
    }
*/
}
