package com.codeberry.tadlib.provider.opencl.ops;

import com.codeberry.tadlib.array.NDIntArray;
import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.array.util.DimensionUtils;
import com.codeberry.tadlib.provider.opencl.OclArray;
import com.codeberry.tadlib.provider.opencl.InProgressResources;
import com.codeberry.tadlib.provider.opencl.OclBuffer;
import com.codeberry.tadlib.provider.opencl.OclIntArray;
import com.codeberry.tadlib.provider.opencl.context.Context;
import com.codeberry.tadlib.provider.opencl.kernel.Kernel;
import com.codeberry.tadlib.provider.opencl.queue.CommandQueue;

import java.util.Collections;
import java.util.List;

import static com.codeberry.tadlib.array.util.DimensionUtils.calcBlockSizes;
import static com.codeberry.tadlib.array.util.DimensionUtils.calcBroadcastBlockSizes;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.ByteSize.sizeOf;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.createBuffer;
import static com.codeberry.tadlib.provider.opencl.OclDataType.*;
import static com.codeberry.tadlib.provider.opencl.buffer.BufferMemFlags.*;
import static com.codeberry.tadlib.provider.opencl.queue.CommandQueue.WorkItemMode.MULTIPLES_OF_PREFERRED_GROUP_SIZE;
import static com.codeberry.tadlib.util.ClassResourceUtils.readString;

public class ArgMax implements OclKernelSource {

    public static final String ARG_MAX = "argMax";

    @Override
    public String getKernelSource() {
        return readString(ArgMax.class, "ArgMax.cl");
    }

    @Override
    public List<String> getKernels() {
        return Collections.singletonList(ARG_MAX);
    }

    public static NDIntArray argMax(Context context, OclArray src, int axis) {
        DimensionUtils.validateAxisWithinBounds(src.getShape(), axis);

        Shape srcShape = src.getShape();
        int axisLen = srcShape.at(axis);
        int safeAxis = srcShape.wrapNegIndex(axis);

        CommandQueue queue = context.getQueue();
        Kernel kernel = context.findKernel(ARG_MAX);

        int maxWorkers = queue.getDevice().getMaxWorkGroupSize();
        int valuesToCheckPerWorker = (axisLen + maxWorkers - 1) / maxWorkers;
        int workersNeeded = (axisLen + valuesToCheckPerWorker - 1) / valuesToCheckPerWorker;
        int actualWorkGroupSize = kernel.alignWithPreferredWorkGroupSizeMultiple(queue, workersNeeded);

        InProgressResources resources = new InProgressResources(context);
        Shape outShape = src.getShape().removeDimAt(axis);
        OclBuffer out = createBuffer(context, sizeOf(cl_int, outShape.getSize()), CL_MEM_READ_WRITE);

        kernel.createArgSetter(resources)
                .nextArg(src)
                .nextArg(srcShape.getSize())
                .nextArg(calcBroadcastBlockSizes(srcShape))
                .nextArg(safeAxis)
                .nextArg(srcShape.at(safeAxis))
                .nextArg(valuesToCheckPerWorker)
                .nextArgKeepRef(out)
                .nextArg(outShape.getSize())
                .nextArg(outShape.getDimCount())
                .nextArg(calcBlockSizes(outShape))
                .nextArgLocalInts(actualWorkGroupSize)
                .nextArgLocalDoubles(actualWorkGroupSize);

        queue.enqueueKernel(kernel,
                new long[]{actualWorkGroupSize, outShape.getSize()},
                new long[]{actualWorkGroupSize, 1},
                resources);

        return OclIntArray.createNDArray(outShape, out, resources);
    }
}
