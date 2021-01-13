package com.codeberry.tadlib.provider.opencl.ops;

import com.codeberry.tadlib.array.NDArray;
import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.opencl.OclArray;
import com.codeberry.tadlib.provider.opencl.OclBuffer;
import com.codeberry.tadlib.provider.opencl.buffer.BufferMemFlags;
import com.codeberry.tadlib.provider.opencl.context.Context;
import com.codeberry.tadlib.provider.opencl.kernel.Kernel;
import com.codeberry.tadlib.provider.opencl.queue.CommandQueue;

import java.util.List;
import java.util.function.Predicate;

import static com.codeberry.tadlib.array.util.DimensionUtils.calcBroadcastBlockSizes;
import static com.codeberry.tadlib.provider.opencl.OclArray.createNDArray;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.ByteSize.sizeOf;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.createBuffer;
import static com.codeberry.tadlib.provider.opencl.OclDataType.cl_double;
import static com.codeberry.tadlib.util.ClassResourceUtils.readString;
import static java.lang.Math.min;
import static java.util.Collections.singletonList;

public class Sum implements OclKernelSource {

    public static final String TENSOR_SUM = "tensorSum";

    @Override
    public String getKernelSource() {
        return readString(Sum.class, "Sum.cl");
    }

    @Override
    public List<String> getKernels() {
        return singletonList(TENSOR_SUM);
    }

    public static NDArray sum(Context context, NDArray src, Boolean[] dimsToSum, NDArray.DimKeepRemove keepRemove) {
        Shape inShape = src.getShape();
        int dimCount = inShape.getDimCount();
        if (dimsToSum.length != dimCount) {
            throw new RuntimeException("dimsToSum must have same length as input shape");
        }

        Shape sumShape = singleDimensionWhen(dimensionToSum -> !dimensionToSum, inShape, dimsToSum, context);
        Shape outShape = singleDimensionWhen(dimensionToSum -> dimensionToSum, inShape, dimsToSum, context);
        long inSize = inShape.getSize();

        CommandQueue queue = context.getQueue();
        Kernel kernel = context.findKernel(TENSOR_SUM);

        long workGroupSizeMultiple = kernel.getPreferredWorkGroupSizeMultiple(queue);

        int neededWorkers = calcWorkGroupSize(sumShape, queue, workGroupSizeMultiple);
        int aggregateValsPerWorker = (int) ((sumShape.getSize() + neededWorkers - 1) / neededWorkers);

        OclBuffer buf = createBuffer(context, sizeOf(cl_double, outShape.getSize()), BufferMemFlags.CL_MEM_READ_WRITE);
        OclArray.InProgressResources resources = new OclArray.InProgressResources(context);

        kernel.createArgSetter(resources)
                .nextArg(dimCount)
                .nextArg((OclArray) src)
                .nextArg(inSize)
                .nextArg(sumShape.getSize())
                .nextArg(calcBroadcastBlockSizes(inShape))
                .nextArg(calcBroadcastBlockSizes(sumShape))
                .nextArg(calcBroadcastBlockSizes(outShape))
                .nextArg(aggregateValsPerWorker)
                .nextArgKeepRef(buf)
                .nextArgLocalDoubles(neededWorkers);


        queue.enqueueKernel(kernel,
                new long[]{neededWorkers, outShape.getSize()},
                new long[]{neededWorkers, 1},
                resources);

        return createNDArray(keepRemove.toActualOutShape(inShape, outShape, dimsToSum), buf, resources);
    }

    private static int calcWorkGroupSize(Shape sumShape, CommandQueue queue, long workGroupSizeMultiple) {
        long maxWorkers = queue.getDevice().info.maxWorkGroupSize.longValue();
        long needWorkers = min((sumShape.getSize() + workGroupSizeMultiple - 1) / workGroupSizeMultiple * workGroupSizeMultiple, maxWorkers);
        return Math.toIntExact(needWorkers);
    }

    private static Shape singleDimensionWhen(Predicate<Boolean> collapseToSingleDimension, Shape inShape, Boolean[] dimsToSum, Context context) {
        int[] dims = inShape.toDimArray();
        for (int i = 0; i < dimsToSum.length; i++) {
            if (collapseToSingleDimension.test(dimsToSum[i])) {
                dims[i] = 1;
            }
        }

        return ProviderStore.shape(dims);
    }
}
