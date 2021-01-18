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
import static com.codeberry.tadlib.array.util.DimensionUtils.validateSameDimensionsExcept;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.ByteSize.sizeOf;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.createBuffer;
import static com.codeberry.tadlib.provider.opencl.OclDataType.cl_double;
import static com.codeberry.tadlib.provider.opencl.buffer.BufferMemFlags.CL_MEM_READ_WRITE;
import static com.codeberry.tadlib.provider.opencl.queue.CommandQueue.WorkItemMode.MULTIPLES_OF_PREFERRED_GROUP_SIZE;
import static com.codeberry.tadlib.util.ClassResourceUtils.readString;
import static java.util.Collections.*;

public class UpdateAtIndicesOnAxis implements OclKernelSource {

    public static final String UPDATE_AT_INDICES_ON_AXIS = "updateAtIndicesOnAxis";

    @Override
    public String getKernelSource() {
        return readString(UpdateAtIndicesOnAxis.class, "UpdateAtIndicesOnAxis.cl");
    }

    @Override
    public List<String> getKernels() {
        return singletonList(UPDATE_AT_INDICES_ON_AXIS);
    }

    public static NDArray updateAtIndicesOnAxis(Context context, OclArray src, OclIntArray indices, int axis, OclArray change) {
        Shape srcShape = src.getShape();
        Shape indicesShape = indices.getShape();

        validateAxisWithinBounds(srcShape, axis);
        validateSameDimensionsExcept("indices", srcShape, indicesShape, axis);
        validateSameDimensionsExcept("change", srcShape, change.getShape(), axis);

        int safeAxis = srcShape.wrapNegIndex(axis);
        int axisLen = srcShape.at(axis);

        CommandQueue queue = context.getQueue();
        Kernel kernel = context.findKernel(UPDATE_AT_INDICES_ON_AXIS);

        Shape outShape = srcShape;
        InProgressResources resources = new InProgressResources(context);
        OclBuffer out = createBuffer(context, sizeOf(cl_double, outShape.getSize()), CL_MEM_READ_WRITE);
        out.oclCopy(queue, resources, src.buffer, 0);

        kernel.createArgSetter(resources)
                .nextArg(outShape.getSize())
                .nextArg(calcBroadcastBlockSizes(outShape))
                .nextArg(safeAxis)
                .nextArg(axisLen)
                .nextArg(indices)
                .nextArg(change)
                .nextArgKeepRef(out)
                .nextArg(indicesShape.getSize())
                .nextArg(indicesShape.getDimCount())
                .nextArg(calcBlockSizes(indicesShape));

        queue.enqueueKernel(kernel,
                indicesShape.getSize(), MULTIPLES_OF_PREFERRED_GROUP_SIZE,
                resources);

        return OclArray.createNDArray(outShape, out, resources);
    }
}
