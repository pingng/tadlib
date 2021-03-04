package com.codeberry.tadlib.provider.opencl.ops;

import com.codeberry.tadlib.array.NDArray;
import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.provider.opencl.InProgressResources;
import com.codeberry.tadlib.provider.opencl.OclArray;
import com.codeberry.tadlib.provider.opencl.OclBuffer;
import com.codeberry.tadlib.provider.opencl.buffer.BufferMemFlags;
import com.codeberry.tadlib.provider.opencl.context.Context;
import com.codeberry.tadlib.provider.opencl.kernel.Kernel;
import com.codeberry.tadlib.provider.opencl.queue.CommandQueue;

import java.util.List;

import static com.codeberry.tadlib.array.util.DimensionUtils.*;
import static com.codeberry.tadlib.array.util.DimensionUtils.evalConcatShape;
import static com.codeberry.tadlib.provider.opencl.OclArray.createNDArray;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.ByteSize.sizeOf;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.createBuffer;
import static com.codeberry.tadlib.provider.opencl.OclDataType.cl_double;
import static com.codeberry.tadlib.util.ClassResourceUtils.readString;
import static java.util.Collections.singletonList;

public class Concat implements OclKernelSource {

    public static final String CONCAT = "concat";
    public static final int MAX_SOURCES = 10;

    @Override
    public String getKernelSource() {
        return readString(Concat.class, "Concat.cl");
    }

    @Override
    public List<String> getKernels() {
        return singletonList(CONCAT);
    }

    public static NDArray concat(Context context, OclArray[] srcs, int axis) {
        if (srcs.length > MAX_SOURCES) {
            throw new IllegalArgumentException("Too many sources, max sources is: " + MAX_SOURCES);
        }
        Shape[] shapes = extractShapes(srcs);
        validateConcatShapes(shapes, axis);
        Shape outShape = evalConcatShape(shapes, axis);

        CommandQueue queue = context.getQueue();
        Kernel kernel = context.findKernel(CONCAT);

        InProgressResources resources = new InProgressResources(context);
        OclBuffer out = createBuffer(context, sizeOf(cl_double, outShape.getSize()), BufferMemFlags.CL_MEM_READ_WRITE);

        Kernel.ArgSetter argSetter = kernel.createArgSetter(resources);
        for (OclArray src : srcs) {
            argSetter.nextArg(src);
        }
        for (int i = 0; i < MAX_SOURCES - srcs.length; i++) {
            argSetter.nextArg((OclArray) null);
        }
        argSetter
                .nextArg(srcs.length)
                .nextArg(axis)
                .nextArg(extractAxisLen(shapes, axis))
                .nextArgKeepRef(out)
                .nextArg(outShape.getDimCount())
                .nextArg(outShape.toDimArray())
                .nextArg(outShape.getSize());

        queue.enqueueKernel(kernel,
                outShape.getSize(), CommandQueue.WorkItemMode.MULTIPLES_OF_PREFERRED_GROUP_SIZE,
                resources);

        return createNDArray(outShape, out, resources);
    }
}
