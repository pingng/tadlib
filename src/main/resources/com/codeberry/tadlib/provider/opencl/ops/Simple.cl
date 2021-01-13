__kernel void arrayNegate(__global const double *A, long len,
                     __global double *C) {
    size_t i = get_global_id(0);

    if( i < len ) {
        C[i] = -A[i];
    }
}

__kernel void arraySqr(__global const double *A, long len,
                     __global double *C) {
    size_t i = get_global_id(0);

    if( i < len ) {
        C[i] = A[i] * A[i];
    }
}

__kernel void arraySqrt(__global const double *A, long len,
                     __global double *C) {
    size_t i = get_global_id(0);

    if( i < len ) {
        C[i] = sqrt(A[i]);
    }
}

__kernel void arrayPow(__global const double *A, long len,
                     double val,
                     __global double *C) {
    size_t i = get_global_id(0);

    if( i < len ) {
        C[i] = pow(A[i], val);
    }
}

__kernel void rot180(__global const double *src,
                     long srcLen,
                     int dimCount,
                     __global const int * dimSizes,
                     __global const int * blockSizes,
                     int yAxis,
                     int xAxis,
                     __global double *out) {

    size_t srcOffset = get_global_id(0);

    if( srcOffset < srcLen ) {
        double val = src[srcOffset];

        long currentOffset = srcOffset;
        long outOffset = 0 ;
        for(int i=0; i<dimCount; i++) {
            int blockSize = blockSizes[i];
            long index = currentOffset / blockSize;

            if(i == yAxis || i == xAxis) {
                index = dimSizes[i] - 1 - index;
            }

            outOffset += index * blockSize;
            currentOffset %= blockSize;
        }

        out[outOffset] = val;
    }
}
