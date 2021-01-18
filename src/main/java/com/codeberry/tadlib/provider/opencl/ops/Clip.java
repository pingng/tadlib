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

import static com.codeberry.tadlib.provider.opencl.OclArray.createNDArray;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.ByteSize.sizeOf;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.createBuffer;
import static com.codeberry.tadlib.provider.opencl.OclDataType.cl_double;
import static com.codeberry.tadlib.provider.opencl.queue.CommandQueue.WorkItemMode.MULTIPLES_OF_PREFERRED_GROUP_SIZE;
import static com.codeberry.tadlib.util.ClassResourceUtils.readString;
import static java.util.Collections.singletonList;

public class Clip implements OclKernelSource {

    public static final String CLIP = "clip";

    @Override
    public String getKernelSource() {
        return readString(Simple.class, "Clip.cl");
    }

    @Override
    public List<String> getKernels() {
        return singletonList(CLIP);
    }

    public static NDArray clip(Context context, OclArray src, Double min, Double max) {
        Shape shape = src.getShape();
        long size = shape.getSize();

        OclBuffer buf = createBuffer(context, sizeOf(cl_double, shape.getSize()), BufferMemFlags.CL_MEM_READ_WRITE);
        InProgressResources resources = new InProgressResources(context);

        Kernel kernel = context.findKernel(CLIP);
        kernel.createArgSetter(resources)
                .nextArg(src)
                .nextArg(size)
                .nextArg(min != null)
                .nextArg(max != null)
                .nextArg((min != null ? min : -1.0))
                .nextArg((max != null ? max : -1.0))
                .nextArgKeepRef(buf);

        CommandQueue queue = context.getQueue();

        queue.enqueueKernel(kernel, size, MULTIPLES_OF_PREFERRED_GROUP_SIZE, resources);

        return createNDArray(shape, buf, resources);
    }
}
