package com.codeberry.tadlib.provider.opencl.ops;

import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.opencl.OclArray;
import com.codeberry.tadlib.provider.opencl.OclBuffer;
import com.codeberry.tadlib.provider.opencl.buffer.BufferMemFlags;
import com.codeberry.tadlib.provider.opencl.context.Context;
import com.codeberry.tadlib.provider.opencl.kernel.Kernel;
import com.codeberry.tadlib.provider.opencl.queue.CommandQueue;

import java.util.List;

import static com.codeberry.tadlib.array.util.DimensionUtils.*;
import static com.codeberry.tadlib.provider.opencl.OclArray.InProgressResources;
import static com.codeberry.tadlib.provider.opencl.OclArray.createNDArray;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.ByteSize.sizeOf;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.createBuffer;
import static com.codeberry.tadlib.provider.opencl.OclDataType.cl_double;
import static com.codeberry.tadlib.provider.opencl.queue.CommandQueue.WorkItemMode.MULTIPLES_OF_PREFERRED_GROUP_SIZE;
import static com.codeberry.tadlib.util.ClassResourceUtils.readString;
import static java.util.Arrays.asList;

public class Mul implements OclKernelSource {

    public static final String TENSOR_MUL_SCALAR = "tensorMulScalar";
    public static final String SINGLE_ELEMENT_MUL = "singleElementMul";
    public static final String ARRAY_MUL = "arrayMul";
    public static final String TENSOR_MUL = "tensorMul";

    @Override
    public String getKernelSource() {
        return readString(Mul.class, "Mul.cl");
    }

    @Override
    public List<String> getKernels() {
        return asList(SINGLE_ELEMENT_MUL, TENSOR_MUL_SCALAR, ARRAY_MUL, TENSOR_MUL);
    }

    public static OclArray mul(Context context, OclArray left, OclArray right) {
        Shape leftShape = left.getShape();
        Shape rightShape = right.getShape();

        if (leftShape.getDimCount() == 0 && rightShape.getDimCount() == 0) {
            return mulScalar(context, left, right);
        } else if (leftShape.equals(rightShape)) {
            return mulSameShapes(context, left, right);
        } else {
            return mulDifferentShapes(context, left, right);
        }
    }

    public static OclArray mul(Context context, OclArray left, double value) {
        Shape leftShape = left.getShape();
        long size = leftShape.getSize();

        OclBuffer buf = createBuffer(context, sizeOf(cl_double, leftShape.getSize()), BufferMemFlags.CL_MEM_READ_WRITE);
        InProgressResources resources = new InProgressResources(context);

        Kernel kernel = context.findKernel(TENSOR_MUL_SCALAR);
        kernel.createArgSetter(resources)
                .nextArg(left)
                .nextArg(value)
                .nextArg(size)
                .nextArgKeepRef(buf);

        CommandQueue queue = context.getQueue();

        queue.enqueueKernel(kernel, size, MULTIPLES_OF_PREFERRED_GROUP_SIZE, resources);

        return createNDArray(leftShape, buf, resources);
    }

    private static OclArray mulDifferentShapes(Context context, OclArray left, OclArray right) {
        Shape leftShape = left.getShape();
        Shape rightShape = right.getShape();

        validateBroadcastShapes(leftShape, rightShape, -1);
        Shape outShape = ProviderStore.shape(createBroadcastResultDims(leftShape, rightShape));
        int outputDimCount = outShape.getDimCount();
        long outSize = outShape.getSize();


        OclBuffer buf = createBuffer(context, sizeOf(cl_double, outShape.getSize()), BufferMemFlags.CL_MEM_READ_WRITE);
        InProgressResources resources = new InProgressResources(context);

        Kernel kernel = context.findKernel(TENSOR_MUL);

        int[] leftBlocks = calcBroadcastBlockSizes(leftShape, outputDimCount);
        int[] rightBlocks = calcBroadcastBlockSizes(rightShape, outputDimCount);
//        System.out.println("leftBlocks = " + Arrays.toString(leftBlocks));
//        System.out.println("rightBlocks = " + Arrays.toString(rightBlocks));
        kernel.createArgSetter(resources)
                .nextArg(left)
                .nextArg(right)
                .nextArg(leftBlocks)
                .nextArg(rightBlocks)
                .nextArg(calcBlockSizes(outShape))
                .nextArg(outputDimCount)
                .nextArgKeepRef(buf)
                .nextArg(outSize);

        CommandQueue queue = context.getQueue();
        queue.enqueueKernel(kernel, outSize, MULTIPLES_OF_PREFERRED_GROUP_SIZE, resources);

//        throwOnError(() -> inst.clWaitForEvents(1, OpenCL.PointerArray.pointers(opEvent.getValue())));

        return createNDArray(outShape, buf, resources);
    }

    private static OclArray mulSameShapes(Context context, OclArray left, OclArray right) {
        Shape leftShape = left.getShape();
        long size = leftShape.getSize();

        OclBuffer buf = createBuffer(context, sizeOf(cl_double, leftShape.getSize()), BufferMemFlags.CL_MEM_READ_WRITE);
        InProgressResources resources = new InProgressResources(context);

        Kernel kernel = context.findKernel(ARRAY_MUL);
        kernel.createArgSetter(resources)
                .nextArg(left)
                .nextArg(right)
                .nextArg(size)
                .nextArgKeepRef(buf);

        CommandQueue queue = context.getQueue();
//        long workItemSize = queue.getDevice().getWorkItemSize(0);
//        SizeTByReference global_item_size;
//        SizeTByReference local_item_size;
//        if (size <= workItemSize) {
//            global_item_size = new SizeTByReference(size);
//            local_item_size = new SizeTByReference(size);
//        } else {
//            long groups = (size + workItemSize - 1) / workItemSize;
//            global_item_size = new SizeTByReference(workItemSize * groups);
//            local_item_size = new SizeTByReference(workItemSize);
//        }
//
//        throwOnError(() -> inst.clEnqueueNDRangeKernel(queue, kernel, 1, null,
//                global_item_size, local_item_size, 0, null, resources.opEvent));
        queue.enqueueKernel(kernel, size, MULTIPLES_OF_PREFERRED_GROUP_SIZE, resources);

        return createNDArray(leftShape, buf, resources);
    }

    private static OclArray mulScalar(Context context, OclArray left, OclArray right) {
        OclBuffer buf = createBuffer(context, sizeOf(cl_double, left.shape.getSize()), BufferMemFlags.CL_MEM_READ_WRITE);
        InProgressResources resources = new InProgressResources(context);

        Kernel kernel = context.findKernel(SINGLE_ELEMENT_MUL);

        kernel.createArgSetter(resources)
                .nextArg(left)
                .nextArg(right)
                .nextArgKeepRef(buf);
        context.getQueue().enqueueKernel(kernel, 1, MULTIPLES_OF_PREFERRED_GROUP_SIZE, resources);

        return createNDArray(left.shape, buf, resources);
    }
}
