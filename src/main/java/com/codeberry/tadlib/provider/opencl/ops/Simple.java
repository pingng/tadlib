package com.codeberry.tadlib.provider.opencl.ops;

import com.codeberry.tadlib.array.NDArray;
import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.provider.opencl.OclArray;
import com.codeberry.tadlib.provider.opencl.OclBuffer;
import com.codeberry.tadlib.provider.opencl.buffer.BufferMemFlags;
import com.codeberry.tadlib.provider.opencl.context.Context;
import com.codeberry.tadlib.provider.opencl.kernel.Kernel;
import com.codeberry.tadlib.provider.opencl.queue.CommandQueue;

import java.util.List;

import static com.codeberry.tadlib.array.util.DimensionUtils.calcBlockSizes;

import com.codeberry.tadlib.provider.opencl.InProgressResources;
import static com.codeberry.tadlib.provider.opencl.OclArray.createNDArray;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.ByteSize.sizeOf;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.createBuffer;
import static com.codeberry.tadlib.provider.opencl.OclDataType.cl_double;
import static com.codeberry.tadlib.provider.opencl.queue.CommandQueue.WorkItemMode.MULTIPLES_OF_PREFERRED_GROUP_SIZE;
import static com.codeberry.tadlib.util.ClassResourceUtils.readString;
import static java.util.Arrays.asList;

public class Simple implements OclKernelSource {

    public static final String ARRAY_NEGATE = "arrayNegate";
    public static final String ARRAY_SQR = "arraySqr";
    public static final String ARRAY_SQRT = "arraySqrt";
    public static final String ARRAY_LOG = "arrayLog";
    public static final String ROT_180 = "rot180";
    public static final String ARRAY_POW = "arrayPow";

    @Override
    public String getKernelSource() {
        return readString(Simple.class, "Simple.cl");
    }

    @Override
    public List<String> getKernels() {
        return asList(ARRAY_NEGATE, ARRAY_SQR, ARRAY_SQRT, ARRAY_LOG, ROT_180, ARRAY_POW);
    }

    public static NDArray log(Context context, OclArray src) {
        return runKernel(context, src, ARRAY_LOG);
    }

    public static OclArray negate(Context context, OclArray src) {
        return runKernel(context, src, ARRAY_NEGATE);
    }

    public static OclArray sqr(Context context, OclArray src) {
        return runKernel(context, src, ARRAY_SQR);
    }

    public static OclArray sqrt(Context context, OclArray src) {
        return runKernel(context, src, ARRAY_SQRT);
    }

    private static OclArray runKernel(Context context, OclArray src, String kernelName) {
        Shape shape = src.getShape();
        long size = shape.getSize();

        OclBuffer buf = createBuffer(context, sizeOf(cl_double, shape.getSize()), BufferMemFlags.CL_MEM_READ_WRITE);
        InProgressResources resources = new InProgressResources(context);

        Kernel kernel = context.findKernel(kernelName);
        kernel.createArgSetter(resources)
                .nextArg(src)
                .nextArg(size)
                .nextArgKeepRef(buf);

        CommandQueue queue = context.getQueue();

        queue.enqueueKernel(kernel, size, MULTIPLES_OF_PREFERRED_GROUP_SIZE, resources);

        return createNDArray(shape, buf, resources);
    }

    public static OclArray pow(Context context, OclArray src, double val) {
        Shape shape = src.getShape();
        long size = shape.getSize();

        OclBuffer buf = createBuffer(context, sizeOf(cl_double, shape.getSize()), BufferMemFlags.CL_MEM_READ_WRITE);
        InProgressResources resources = new InProgressResources(context);

        Kernel kernel = context.findKernel(ARRAY_POW);
        kernel.createArgSetter(resources)
                .nextArg(src)
                .nextArg(size)
                .nextArg(val)
                .nextArgKeepRef(buf);

        CommandQueue queue = context.getQueue();

        queue.enqueueKernel(kernel, size, MULTIPLES_OF_PREFERRED_GROUP_SIZE, resources);

        return createNDArray(shape, buf, resources);
    }

    public static NDArray rot180(Context context, OclArray src, int yAxis, int xAxis) {
        Shape shape = src.getShape();
        if (shape.getDimCount() < 2) {
            throw new RuntimeException("Must have at least 2 dimension: dimCount=" + shape.getDimCount());
        }

        long size = shape.getSize();

        OclBuffer buf = createBuffer(context, sizeOf(cl_double, shape.getSize()), BufferMemFlags.CL_MEM_READ_WRITE);
        InProgressResources resources = new InProgressResources(context);

        Kernel kernel = context.findKernel(ROT_180);
        kernel.createArgSetter(resources)
                .nextArg(src)
                .nextArg(size)
                .nextArg(shape.getDimCount())
                .nextArg(shape.toDimArray())
                .nextArg(calcBlockSizes(shape))
                .nextArg(yAxis)
                .nextArg(xAxis)
                .nextArgKeepRef(buf);

        CommandQueue queue = context.getQueue();

        queue.enqueueKernel(kernel, size, MULTIPLES_OF_PREFERRED_GROUP_SIZE, resources);

        return createNDArray(shape, buf, resources);
    }

}
