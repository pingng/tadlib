package com.codeberry.tadlib.provider.opencl;

// From: https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/scalarDataTypes.html
public enum OclDataType {
    //bool	A conditional data type which is either true or false. The value true expands to the integer constant 1 and the value false expands to the integer constant 0.

    //char	A signed two's complement 8-bit integer.
    cl_char(1, true),

    //unsigned char, uchar An unsigned 8-bit integer.
    cl_uchar(1, false),

    //short	A signed two's complement 16-bit integer.
    cl_short(2, true),

    //unsigned short, ushort An unsigned 16-bit integer.
    cl_ushort(2, false),

    //int	A signed two's complement 32-bit integer.
    cl_int(4, true),

    //unsigned int, uint An unsigned 32-bit integer.
    cl_uint(4, false),

    //long	A signed two's complement 64-bit integer.
    cl_long(8, true),

    //unsigned long, ulong An unsigned 64-bit integer.
    cl_ulong(8, false),

    //float	A single precision float. The float data type must conform to the IEEE 754 single precision storage format.
    cl_float(4, true),

    //half	A 16-bit float. The half data type must conform to the IEEE 754-2008 half precision storage format.
    cl_half(2, true),

/*
    size_t	The unsigned integer type of the result of the sizeof operator. This is a 32-bit unsigned integer if CL_DEVICE_ADDRESS_BITS defined in clGetDeviceInfo is 32-bits and is a 64-bit unsigned integer if CL_DEVICE_ADDRESS_BITS is 64-bits.	n/a
    ptrdiff_t	A signed integer type that is the result of subtracting two pointers. This is a 32-bit signed integer if CL_DEVICE_ADDRESS_BITS defined in clGetDeviceInfo is 32-bits and is a 64-bit signed integer if CL_DEVICE_ADDRESS_BITS is 64-bits.	n/a
    intptr_t	A signed integer type with the property that any valid pointer to void can be converted to this type, then converted back to pointer to void, and the result will compare equal to the original pointer.	n/a
    uintptr_t	An unsigned integer type with the property that any valid pointer to void can be converted to this type, then converted back to pointer to void, and the result will compare equal to the original pointer.	n/a
    void	The void type comprises an empty set of values; it is an incomplete type that cannot be completed.	void
*/
    //double	A double precision float.
    cl_double(8, true);

    final int byteSize;
    final boolean signed;

    OclDataType(int byteSize, boolean signed) {
        this.byteSize = byteSize;
        this.signed = signed;
    }

    public long sizeOfElements(long elementCount) {
        return elementCount * byteSize;
    }
}
