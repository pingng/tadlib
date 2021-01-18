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

import static com.codeberry.tadlib.array.util.DimensionUtils.getMaxPool2dResultShape;
import static com.codeberry.tadlib.provider.opencl.OclArray.createNDArray;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.ByteSize.sizeOf;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.createBuffer;
import static com.codeberry.tadlib.provider.opencl.OclDataType.cl_double;
import static com.codeberry.tadlib.provider.opencl.OclDataType.cl_int;
import static com.codeberry.tadlib.util.ClassResourceUtils.readString;
import static java.lang.Math.*;
import static java.lang.Math.min;
import static java.util.Arrays.asList;

public class MaxPool implements OclKernelSource {

    public static final String MAX_POOL_2D = "maxPool2d";
    public static final String MAX_POOL_2D_REVERT= "maxPool2dRevert";

    @Override
    public String getKernelSource() {
        return readString(MaxPool.class, "MaxPool.cl");
    }

    @Override
    public List<String> getKernels() {
        return asList(MAX_POOL_2D, MAX_POOL_2D_REVERT);
    }

    public static NDArray.MaxPool2dResult maxPool2d(Context context, OclArray src, int maxPoolSize) {
        Shape srcShape = src.getShape();
        Shape outShape = getMaxPool2dResultShape(srcShape, maxPoolSize);

        InProgressResources resources = new InProgressResources(context);
        OclBuffer buf = createBuffer(context, sizeOf(cl_double, outShape.getSize()), BufferMemFlags.CL_MEM_READ_WRITE);
        OclBuffer bufId = createBuffer(context, sizeOf(cl_int, outShape.getSize()), BufferMemFlags.CL_MEM_READ_WRITE);

        CommandQueue queue = context.getQueue();
        Kernel kernel = context.findKernel(MAX_POOL_2D);

        int inputsPerOutputValue = maxPoolSize * maxPoolSize;
        long outputValues = outShape.getSize();
        long maxWorkers = queue.getDevice().info.maxWorkGroupSize.longValue();

        if (inputsPerOutputValue > maxWorkers) {
            throw new RuntimeException("Kernel supports only " + maxWorkers + " inputs for output. Please rewrite the kernel.");
        }

        long workGroupSizeMultiple = kernel.getPreferredWorkGroupSizeMultiple(queue);
        int neededWorkers = toMultiplesOf(inputsPerOutputValue, maxWorkers, workGroupSizeMultiple);
//        System.out.println("MaxPool: Size="+srcShape.getSize());
        int channels = srcShape.at(-1);
        int src2dArea = toIntExact(srcShape.mulDims(-3, -1));
        int src2dWidth = srcShape.at(-2);
        int src2dHeight = srcShape.at(-3);
        int out2dArea = toIntExact(outShape.mulDims(-3, -1));
        int out2dWidth = outShape.at(-2);
        kernel.createArgSetter(resources)
                .nextArg(maxPoolSize)
                .nextArg(channels)
                .nextArg(src)
                .nextArg(src2dArea)
                .nextArg(src2dWidth)
                .nextArg(src2dHeight)
                .nextArg(inputsPerOutputValue)
                .nextArg(out2dArea)
                .nextArg(out2dWidth)
                .nextArgKeepRef(buf)
                .nextArgKeepRef(bufId)
                .nextArgLocalDoubles(neededWorkers)
                .nextArgLocalInts(neededWorkers);

        queue.enqueueKernel(kernel,
                new long[]{neededWorkers, outputValues},
                new long[]{neededWorkers, 1},
                resources);

        // BufId is needed (probably multiple times) during backprop,
        // but will be disposed at end of iteration
        DisposalRegister.registerForDisposalAtEndOfModelIteration(bufId);

        OclArray output = createNDArray(outShape, buf, resources);

        return new OpenClMaxPool2dResult(output, srcShape, maxPoolSize, bufId);
    }

    private static int toMultiplesOf(int minValue, long maxValue, long multiples) {
        return toIntExact(min((minValue + multiples - 1) / multiples * multiples, maxValue));
    }

    public static NDArray maxPool2dGrad(Context context, OclArray gradSrc, NDArray.MaxPool2dResult result) {
        OpenClMaxPool2dResult r = (OpenClMaxPool2dResult) result;

        Shape outShape = r.orgShape;
        Shape gradShape = gradSrc.getShape();

        InProgressResources resources = new InProgressResources(context);

        CommandQueue queue = context.getQueue();
        Kernel kernel = context.findKernel(MAX_POOL_2D_REVERT);

        OclBuffer orgOut = createBuffer(context, sizeOf(cl_double, outShape.getSize()), BufferMemFlags.CL_MEM_READ_WRITE);

        // Must clear buffer, since kernel will not write to all locations
        orgOut.oclFill(queue, resources, 0);

        int grad2dArea = toIntExact(gradShape.mulDims(-3, -1));
        int grad2dWidth = gradShape.at(-2);
        //int neededWorkers = toMultiplesOf(grad2dArea, maxWorkers, workGroupSizeMultiple);
        int org2dArea = toIntExact(outShape.mulDims(-3, -1));
        int org2dWidth = outShape.at(-2);
        int channels = gradShape.at(-1);

        kernel.createArgSetter(resources)
                .nextArg(r.maxPoolSize)
                .nextArg(channels)
                .nextArg(gradSrc)
                .nextArg(gradShape.getSize())
                .nextArg(grad2dArea)
                .nextArg(grad2dWidth)
                .nextArgKeepRef(r.bufId)
                .nextArg(org2dArea)
                .nextArg(org2dWidth)
                .nextArgKeepRef(orgOut);

        queue.enqueueKernel(kernel,
                gradShape.getSize(), CommandQueue.WorkItemMode.MULTIPLES_OF_PREFERRED_GROUP_SIZE,
                resources);

        return createNDArray(outShape, orgOut, resources);
    }

    private static class OpenClMaxPool2dResult implements NDArray.MaxPool2dResult {
        private final OclArray output;
        private final Shape orgShape;
        private final int maxPoolSize;
        private final OclBuffer bufId;

        public OpenClMaxPool2dResult(OclArray output, Shape orgShape, int maxPoolSize, OclBuffer bufId) {
            this.output = output;
            this.orgShape = orgShape;
            this.maxPoolSize = maxPoolSize;
            this.bufId = bufId;
        }

        @Override
        public NDArray getOutput() {
            return output;
        }
    }
}
