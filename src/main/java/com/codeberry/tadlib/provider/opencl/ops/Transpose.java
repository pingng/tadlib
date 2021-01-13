package com.codeberry.tadlib.provider.opencl.ops;

import com.codeberry.tadlib.array.NDArray;
import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.opencl.OclArray;
import com.codeberry.tadlib.provider.opencl.OclArray.InProgressResources;
import com.codeberry.tadlib.provider.opencl.OclBuffer;
import com.codeberry.tadlib.provider.opencl.context.Context;
import com.codeberry.tadlib.provider.opencl.kernel.Kernel;
import com.codeberry.tadlib.provider.opencl.queue.CommandQueue;

import java.util.List;

import static com.codeberry.tadlib.array.util.DimensionUtils.*;
import static com.codeberry.tadlib.provider.opencl.OclArray.createNDArray;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.ByteSize.sizeOf;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.createBuffer;
import static com.codeberry.tadlib.provider.opencl.OclDataType.cl_double;
import static com.codeberry.tadlib.provider.opencl.buffer.BufferMemFlags.CL_MEM_READ_WRITE;
import static com.codeberry.tadlib.util.ClassResourceUtils.readString;
import static java.util.Collections.singletonList;

public class Transpose implements OclKernelSource {

    public static final String TRANSPOSE = "transpose";

    @Override
    public String getKernelSource() {
        return readString(Transpose.class, "Transpose.cl");
    }

    @Override
    public List<String> getKernels() {
        return singletonList(TRANSPOSE);
    }

    public static NDArray transpose(Context context, OclArray src, int[] _axes) {
        int[] internAxes = (_axes.length == 0 ? reverseArray(src.getShape().getDimCount()) : _axes);
        Shape srcShape = src.getShape();
        validateTransposeAxes(srcShape, internAxes);
        int dimCount = srcShape.getDimCount();

        int[] dims = new int[dimCount];
        for (int i = 0; i < dims.length; i++) {
            dims[i] = srcShape.at(internAxes[i]);
        }
        Shape outShape = ProviderStore.shape(dims);

        if (srcShape.getSize() != outShape.getSize()) {
            throw new RuntimeException("sizes should be equal!");
        }

        OclBuffer buf = createBuffer(context, sizeOf(cl_double, outShape.getSize()), CL_MEM_READ_WRITE);

        CommandQueue queue = context.getQueue();
        Kernel kernel = context.findKernel(TRANSPOSE);
        InProgressResources resources = new InProgressResources(context);

        long workGroupSizeMultiple = kernel.getPreferredWorkGroupSizeMultiple(queue);
        WorkGroupRect workGroupRect = WorkGroupRect.evalMostSquareRect((int) workGroupSizeMultiple);

        int[] srcBlockSizes = calcBlockSizes(srcShape);
        int[] orgOutBroadcastBlockSizes = calcBroadcastBlockSizes(outShape);
        int[] outBroadcastBlockSizes = new int[orgOutBroadcastBlockSizes.length];
        for (int i = 0; i < outBroadcastBlockSizes.length; i++) {
            outBroadcastBlockSizes[internAxes[i]] = orgOutBroadcastBlockSizes[i];
        }

        int workgroupSize = workGroupRect.height * workGroupRect.width;

        kernel.createArgSetter(resources)
                .nextArg(src)
                .nextArg(srcBlockSizes)
                .nextArgKeepRef(buf)
                .nextArg(outBroadcastBlockSizes)
                .nextArg(outShape.getDimCount())
                .nextArg(outShape.getSize())
                .nextArg(workGroupRect.width)
                .nextArgLocalDoubles(workgroupSize);

        queue.enqueueKernel(kernel,
                srcShape.getSize(), CommandQueue.WorkItemMode.MULTIPLES_OF_PREFERRED_GROUP_SIZE,
                resources);

        return createNDArray(outShape, buf, resources);
    }

    private static int[] reverseArray(int length) {
        int[] r = new int[length];
        for (int i = 0; i < r.length; i++) {
            r[i] = length - i - 1;
        }
        return r;
    }

    private static class WorkGroupRect {
        final int height;
        final int width;

        private WorkGroupRect(int height, int width) {
            this.height = height;
            this.width = width;
        }

        @Override
        public String toString() {
            return "WorkGroupRect{" +
                    "height=" + height +
                    ", width=" + width +
                    '}';
        }

        static WorkGroupRect evalMostSquareRect(int totalSize) {
            int width = (int) Math.sqrt(totalSize);
            while (width < totalSize && (totalSize % width) != 0) {
                width++;
            }
            return new WorkGroupRect(totalSize / width, width);
        }
    }
}
