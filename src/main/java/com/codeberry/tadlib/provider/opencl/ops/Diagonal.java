package com.codeberry.tadlib.provider.opencl.ops;

import com.codeberry.tadlib.array.NDArray;
import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.array.util.DimensionUtils;
import com.codeberry.tadlib.memorymanagement.DisposalRegister;
import com.codeberry.tadlib.provider.opencl.InProgressResources;
import com.codeberry.tadlib.provider.opencl.OclArray;
import com.codeberry.tadlib.provider.opencl.OclBuffer;
import com.codeberry.tadlib.provider.opencl.buffer.BufferMemFlags;
import com.codeberry.tadlib.provider.opencl.context.Context;
import com.codeberry.tadlib.provider.opencl.kernel.Kernel;
import com.codeberry.tadlib.provider.opencl.queue.CommandQueue;

import java.util.Collections;
import java.util.List;

import static com.codeberry.tadlib.array.util.DimensionUtils.calcBlockSizes;
import static com.codeberry.tadlib.array.util.DimensionUtils.calcBroadcastBlockSizes;
import static com.codeberry.tadlib.provider.opencl.OclArray.createNDArray;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.ByteSize.sizeOf;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.createBuffer;
import static com.codeberry.tadlib.provider.opencl.OclDataType.cl_double;
import static com.codeberry.tadlib.util.ClassResourceUtils.readString;
import static java.util.Collections.singletonList;

public class Diagonal implements OclKernelSource {

    public static final String DIAG = "diag";

    @Override
    public String getKernelSource() {
        return readString(Diagonal.class, "Diagonal.cl");
    }

    @Override
    public List<String> getKernels() {
        return singletonList(DIAG);
    }

    public static NDArray diag(Context context, OclArray src) {
        Shape shape = src.getShape();
        Shape outShape = shape.appendDim(shape.at(-1));

        CommandQueue queue = context.getQueue();
        Kernel kernel = context.findKernel(DIAG);

        InProgressResources resources = new InProgressResources(context);
        OclBuffer out = createBuffer(context, sizeOf(cl_double, outShape.getSize()), BufferMemFlags.CL_MEM_READ_WRITE);

        out.oclFill(queue, resources, 0.0);

        kernel.createArgSetter(resources)
                .nextArg(src)
                .nextArg(shape.getDimCount())
                .nextArg(calcBlockSizes(shape))
                .nextArg(shape.getSize())
                .nextArgKeepRef(out)
                .nextArg(calcBroadcastBlockSizes(outShape));

        queue.enqueueKernel(kernel,
                shape.getSize(), CommandQueue.WorkItemMode.MULTIPLES_OF_PREFERRED_GROUP_SIZE,
                resources);

        return createNDArray(outShape, out, resources);
    }
}
