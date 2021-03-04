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

import java.util.ArrayList;
import java.util.List;

import static com.codeberry.tadlib.array.util.DimensionUtils.*;
import static com.codeberry.tadlib.provider.opencl.OclArray.createNDArray;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.ByteSize.sizeOf;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.createBuffer;
import static com.codeberry.tadlib.provider.opencl.OclDataType.cl_double;
import static com.codeberry.tadlib.util.ClassResourceUtils.readString;
import static java.util.Collections.singletonList;

public class Split implements OclKernelSource {

    public static final String SPLIT = "split";
    public static final int MAX_TARGETS = 10;

    @Override
    public String getKernelSource() {
        return readString(Split.class, "Split.cl");
    }

    @Override
    public List<String> getKernels() {
        return singletonList(SPLIT);
    }

    public static List<NDArray> split(Context context, OclArray src, int axis, int[] axisLens) {
        if (axisLens.length > MAX_TARGETS) {
            throw new IllegalArgumentException("Too many targets, max targets is: " + MAX_TARGETS);
        }
        Shape srcShape = src.getShape();
        validateSplitLens(srcShape, axis, axisLens);

        CommandQueue queue = context.getQueue();
        Kernel kernel = context.findKernel(SPLIT);

        List<Shape> outShapes = createOutShapes(axis, axisLens, srcShape);
        List<OclBuffer> outBuffers = createBuffers(context, outShapes);

        InProgressResources resources = new InProgressResources(context);

        Kernel.ArgSetter argSetter = kernel.createArgSetter(resources);
        argSetter
                .nextArg(src)
                .nextArg(srcShape.getDimCount())
                .nextArg(srcShape.toDimArray())
                .nextArg(srcShape.getSize())
                .nextArg(axis)
                .nextArg(axisLens);
        for (OclBuffer outBuffer : outBuffers) {
            argSetter.nextArgKeepRef(outBuffer);
        }
        for (int i = 0; i < MAX_TARGETS - axisLens.length; i++) {
            argSetter.nextArgKeepRef(null);
        }

        queue.enqueueKernel(kernel,
                srcShape.getSize(), CommandQueue.WorkItemMode.MULTIPLES_OF_PREFERRED_GROUP_SIZE,
                resources);

        List<NDArray> outArrays = new ArrayList<>();
        for (int i = 0; i < outShapes.size(); i++) {
            Shape shape = outShapes.get(i);
            OclBuffer buffer = outBuffers.get(i);
            outArrays.add(createNDArray(shape, buffer, resources));
        }
        return outArrays;
    }

    private static List<OclBuffer> createBuffers(Context context, List<Shape> outShapes) {
        List<OclBuffer> outBuffers = new ArrayList<>();
        for (Shape outShape : outShapes) {
            outBuffers.add(createBuffer(context, sizeOf(cl_double, outShape.getSize()), BufferMemFlags.CL_MEM_READ_WRITE));
        }
        return outBuffers;
    }

    private static List<Shape> createOutShapes(int axis, int[] axisLens, Shape srcShape) {
        List<Shape> outShapes = new ArrayList<>();
        for (int axisLen : axisLens) {
            outShapes.add(evalSplitShape(srcShape, axis, axisLen));
        }
        return outShapes;
    }
}
