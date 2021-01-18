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
import static com.codeberry.tadlib.util.ClassResourceUtils.readString;
import static java.util.Collections.singletonList;

public class Update implements OclKernelSource {

    public static final String UPDATE_DOUBLE = "updateDouble";

    @Override
    public String getKernelSource() {
        return readString(Update.class, "Update.cl");
    }

    @Override
    public List<String> getKernels() {
        return singletonList(UPDATE_DOUBLE);
    }

    public static NDArray update(Context context, OclArray src, List<NDArray.ValueUpdate> updates) {
        Shape shape = src.getShape();
        long size = shape.getSize();

        InProgressResources resources = new InProgressResources(context);
        OclBuffer buf = createBuffer(context, sizeOf(cl_double, size), BufferMemFlags.CL_MEM_READ_WRITE);

        CommandQueue queue = context.getQueue();
        buf.oclCopy(queue, resources, src.buffer, 0);

        Kernel kernel = context.findKernel(UPDATE_DOUBLE);

        long[] offsets = new long[updates.size()];
        double[] doubles = new double[updates.size()];
        for (int i = 0; i < updates.size(); i++) {
            NDArray.ValueUpdate update = updates.get(i);
            offsets[i] = update.offset;
            doubles[i] = update.value;
        }

        kernel.createArgSetter(resources)
                .nextArgKeepRef(buf)
                .nextArg(size)
                .nextArg(offsets)
                .nextArg(doubles)
                .nextArg(offsets.length);

        queue.enqueueKernel(kernel,
                updates.size(), CommandQueue.WorkItemMode.MULTIPLES_OF_PREFERRED_GROUP_SIZE,
                resources);

        return createNDArray(shape, buf, resources);
    }
}
