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
import static com.codeberry.tadlib.array.util.DimensionUtils.MatMulParams.expandSingleDimArrays;
import static com.codeberry.tadlib.array.util.DimensionUtils.calcBroadcastBlockSizes;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.ByteSize.sizeOf;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.createBuffer;
import static com.codeberry.tadlib.provider.opencl.OclDataType.*;
import static com.codeberry.tadlib.provider.opencl.buffer.BufferMemFlags.*;
import static com.codeberry.tadlib.util.ClassResourceUtils.readString;
import static java.util.Collections.singletonList;

public class MatMul implements OclKernelSource {

    public static final String MAT_MUL = "matmul";

    @Override
    public String getKernelSource() {
        return readString(MatMul.class, "MatMul.cl");
    }

    @Override
    public List<String> getKernels() {
        return singletonList(MAT_MUL);
    }

    public static NDArray matmul(Context context, OclArray left, OclArray right) {
        MatMulParams params = expandSingleDimArrays(left.getShape(), right.getShape(),
                ProviderStore::shape);

        validateMatMulShapes(params.leftShape, params.rightShape);
        validateBroadcastShapes(params.leftShape, params.rightShape, -3);

        Shape outShape = ProviderStore.shape(evalMatMulResultDims(params.leftShape, params.rightShape));
        OclBuffer buf = createBuffer(context, sizeOf(cl_double, ((Shape) outShape).getSize()), CL_MEM_READ_WRITE);

        CommandQueue queue = context.getQueue();
        Kernel kernel = context.findKernel(MAT_MUL);
        InProgressResources resources = new InProgressResources(context);

        long workGroupSizeMultiple = kernel.getPreferredWorkGroupSizeMultiple(queue);
        WorkGroupRect workGroupRect = WorkGroupRect.evalMostSquareRect((int) workGroupSizeMultiple);

        int mulsPerOutput = params.leftShape.at(-1);

        int[] leftBroadcastBlockSizes = calcBroadcastBlockSizes(params.leftShape, outShape.getDimCount());
        int[] rightBroadcastBlockSizes = calcBroadcastBlockSizes(params.rightShape, outShape.getDimCount());
        int[] outBroadcastBlockSizes = calcBroadcastBlockSizes(outShape, outShape.getDimCount());

        kernel.createArgSetter(resources)
                .nextArg(left)
                .nextArg(left.getShape().getSize())
                .nextArg(right)
                .nextArg(right.getShape().getSize())
                .nextArg(leftBroadcastBlockSizes)
                .nextArg(rightBroadcastBlockSizes)
                .nextArg(mulsPerOutput)
                .nextArg(outShape.getDimCount())
                .nextArg(outShape.getDimensions())
                .nextArg(calcBlockSizes(outShape, 0, -2))
                .nextArgKeepRef(buf)
                .nextArg(outBroadcastBlockSizes);

        long totalExampleCount = calcExampleCount(outShape, ShapeEndType.END_WITH__HEIGHT_WIDTH);
        long workSquareWidth = (outShape.at(-1) + workGroupRect.width - 1) / workGroupRect.width;
        long workSquareHeight = (outShape.at(-2) + workGroupRect.height - 1) / workGroupRect.height;

        queue.waitForFinish();
        queue.enqueueKernel(kernel,
                new long[]{workSquareHeight * workGroupRect.height, workSquareWidth * workGroupRect.width, totalExampleCount},
                new long[]{workGroupRect.height, workGroupRect.width, 1},
                resources);

        return OclArray.createNDArray(params.revertDimExpandOfOutputShape(outShape), buf, resources);
    }

}
