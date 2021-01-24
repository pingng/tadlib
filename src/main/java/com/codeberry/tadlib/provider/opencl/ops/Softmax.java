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
import static java.lang.Long.max;
import static java.lang.Math.min;
import static java.lang.Math.toIntExact;
import static java.util.Collections.singletonList;

public class Softmax implements OclKernelSource {

    public static final String SOFTMAX = "softmax";

    @Override
    public String getKernelSource() {
        return readString(Softmax.class, "Softmax.cl");
    }

    @Override
    public List<String> getKernels() {
        return singletonList(SOFTMAX);
    }

    public static NDArray softmax(Context context, OclArray src) {
        Shape shape = src.getShape();

        InProgressResources resources = new InProgressResources(context);
        OclBuffer out = createBuffer(context, sizeOf(cl_double, shape.getSize()), BufferMemFlags.CL_MEM_READ_WRITE);

        CommandQueue queue = context.getQueue();
        Kernel kernel = context.findKernel(SOFTMAX);

        int valuesPerExample = shape.at(-1);
        long maxWorkers = queue.getDevice().info.maxWorkGroupSize.longValue();

        if (valuesPerExample > maxWorkers) {
            throw new RuntimeException("Kernel supports only " + maxWorkers + " values per example. Please rewrite the kernel.");
        }

        long workGroupSizeMultiple = kernel.getPreferredWorkGroupSizeMultiple(queue);
        int neededWorkers = toMultiplesOf(valuesPerExample, maxWorkers, workGroupSizeMultiple);

        long exampleCount = max(shape.mulDims(0, -1), 1);
        kernel.createArgSetter(resources)
                .nextArg(src)
                .nextArg(valuesPerExample)
                .nextArgKeepRef(out)
                .nextArgLocalDoubles(neededWorkers);

        queue.enqueueKernel(kernel,
                new long[]{neededWorkers, exampleCount},
                new long[]{neededWorkers, 1},
                resources);

        return createNDArray(shape, out, resources);
    }

    private static int toMultiplesOf(int minValue, long maxValue, long multiples) {
        return toIntExact(min((minValue + multiples - 1) / multiples * multiples, maxValue));
    }
}
