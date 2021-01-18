package com.codeberry.tadlib.provider.opencl.ops;

import com.codeberry.tadlib.array.NDArray;
import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.memorymanagement.DisposalRegister;
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
import static java.lang.Math.min;
import static java.util.Collections.singletonList;

public class Relu implements OclKernelSource {

    public static final String RELU = "relu";

    @Override
    public String getKernelSource() {
        return readString(Relu.class, "Relu.cl");
    }

    @Override
    public List<String> getKernels() {
        return singletonList(RELU);
    }

    public static NDArray.ReluResult relu(Context context, OclArray src, double leakyScale) {
        Shape shape = src.getShape();

        InProgressResources resources = new InProgressResources(context);
        OclBuffer buf = createBuffer(context, sizeOf(cl_double, shape.getSize()), BufferMemFlags.CL_MEM_READ_WRITE);
        OclBuffer bufMask = createBuffer(context, sizeOf(cl_double, shape.getSize()), BufferMemFlags.CL_MEM_READ_WRITE);

        CommandQueue queue = context.getQueue();
        Kernel kernel = context.findKernel(RELU);

        kernel.createArgSetter(resources)
                .nextArg(src)
                .nextArg(leakyScale)
                .nextArgKeepRef(buf)
                .nextArgKeepRef(bufMask)
                .nextArg(shape.getSize());

        queue.enqueueKernel(kernel,
                shape.getSize(), CommandQueue.WorkItemMode.MULTIPLES_OF_PREFERRED_GROUP_SIZE,
                resources);

        OclArray output = createNDArray(shape, buf, resources);
        OclArray mask = createNDArray(shape, bufMask, resources);

        DisposalRegister.registerForDisposalAtEndOfModelIteration(mask);

        return new NDArray.ReluResult() {
            @Override
            public NDArray getOutput() {
                return output;
            }

            @Override
            public NDArray createMask() {
                return mask;
            }
        };
    }
}
