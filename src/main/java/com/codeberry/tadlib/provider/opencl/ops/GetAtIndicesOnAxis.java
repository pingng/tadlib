package com.codeberry.tadlib.provider.opencl.ops;

import com.codeberry.tadlib.array.NDArray;
import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.provider.opencl.InProgressResources;
import com.codeberry.tadlib.provider.opencl.OclArray;
import com.codeberry.tadlib.provider.opencl.OclBuffer;
import com.codeberry.tadlib.provider.opencl.OclIntArray;
import com.codeberry.tadlib.provider.opencl.context.Context;
import com.codeberry.tadlib.provider.opencl.kernel.Kernel;
import com.codeberry.tadlib.provider.opencl.queue.CommandQueue;

import java.util.Collections;
import java.util.List;

import static com.codeberry.tadlib.array.util.DimensionUtils.*;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.ByteSize.sizeOf;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.createBuffer;
import static com.codeberry.tadlib.provider.opencl.OclDataType.cl_double;
import static com.codeberry.tadlib.provider.opencl.buffer.BufferMemFlags.CL_MEM_READ_WRITE;
import static com.codeberry.tadlib.provider.opencl.queue.CommandQueue.WorkItemMode.MULTIPLES_OF_PREFERRED_GROUP_SIZE;
import static com.codeberry.tadlib.util.ClassResourceUtils.readString;

public class GetAtIndicesOnAxis implements OclKernelSource {

    public static final String GET_INDICES_ON_AXIS = "getIndicesOnAxis";

    @Override
    public String getKernelSource() {
        return readString(GetAtIndicesOnAxis.class, "GetAtIndicesOnAxis.cl");
    }

    @Override
    public List<String> getKernels() {
        return Collections.singletonList(GET_INDICES_ON_AXIS);
    }

    public static NDArray getAtIndicesOnAxis(Context context, OclArray src, OclIntArray indices, int axis) {
        Shape srcShape = src.getShape();

        validateAxisWithinBounds(srcShape, axis);
        validateSameDimensionsExcept("indices", srcShape, indices.getShape(), axis);

        Shape indicesShape = indices.getShape();
        int safeAxis = srcShape.wrapNegIndex(axis);

        CommandQueue queue = context.getQueue();
        Kernel kernel = context.findKernel(GET_INDICES_ON_AXIS);

        Shape outShape = indicesShape;
        InProgressResources resources = new InProgressResources(context);
        OclBuffer out = createBuffer(context, sizeOf(cl_double, outShape.getSize()), CL_MEM_READ_WRITE);

        kernel.createArgSetter(resources)
                .nextArg(src)
                .nextArg(srcShape.getSize())
                .nextArg(calcBroadcastBlockSizes(srcShape))
                .nextArg(safeAxis)
                .nextArg(indices)
                .nextArgKeepRef(out)
                .nextArg(outShape.getSize())
                .nextArg(outShape.getDimCount())
                .nextArg(calcBlockSizes(outShape));

        queue.enqueueKernel(kernel,
                outShape.getSize(), MULTIPLES_OF_PREFERRED_GROUP_SIZE,
                resources);

        return OclArray.createNDArray(outShape, out, resources);
    }
}
