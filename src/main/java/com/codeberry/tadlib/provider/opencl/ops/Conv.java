package com.codeberry.tadlib.provider.opencl.ops;

import com.codeberry.tadlib.array.NDArray;
import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.opencl.InProgressResources;
import com.codeberry.tadlib.provider.opencl.OclArray;
import com.codeberry.tadlib.provider.opencl.OclBuffer;
import com.codeberry.tadlib.provider.opencl.context.Context;
import com.codeberry.tadlib.provider.opencl.device.Device;
import com.codeberry.tadlib.provider.opencl.kernel.Kernel;
import com.codeberry.tadlib.provider.opencl.queue.CommandQueue;

import java.util.List;

import static com.codeberry.tadlib.array.util.DimensionUtils.ShapeEndType.END_WITH__HEIGHT_WIDTH_CHANNEL;
import static com.codeberry.tadlib.array.util.DimensionUtils.calcExampleCount;
import static com.codeberry.tadlib.provider.opencl.OclArray.*;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.ByteSize.sizeOf;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.createBuffer;
import static com.codeberry.tadlib.provider.opencl.OclDataType.cl_double;
import static com.codeberry.tadlib.provider.opencl.buffer.BufferMemFlags.*;
import static com.codeberry.tadlib.provider.opencl.consts.ErrorCode.throwOnError;
import static com.codeberry.tadlib.util.ClassResourceUtils.readString;
import static java.lang.Math.min;
import static java.util.Collections.singletonList;

public class Conv implements OclKernelSource {

    public static final String CONV_2D = "conv2d";

    @Override
    public String getKernelSource() {
        return readString(Conv.class, "Conv.cl");
    }

    @Override
    public List<String> getKernels() {
        return singletonList(CONV_2D);
    }

    public static class ConvSize {
        public final int height;
        public final int width;

        private ConvSize(int height, int width) {
            this.height = height;
            this.width = width;
        }

        public long calcArea() {
            return (long) height * width;
        }

        public static class Builder {
            private int height;
            private int width;

            public static Builder convSizeBuilder() {
                return new Builder();
            }

            public Builder height(int height) {
                this.height = height;
                return this;
            }

            public Builder width(int width) {
                this.width = width;
                return this;
            }

            public ConvSize build() {
                return new ConvSize(height, width);
            }
        }
    }

    public static OclArray conv2d(Context context, OclArray input, OclArray filter,
                                 int[] filterOffsets,
                                 ConvSize outputSize) {
        NDArray.validateConv2dShapes(input.shape, filter.shape);

        Shape outShape = evalOutputShape(context, input.shape, filter.shape, outputSize);

        InProgressResources resources = new InProgressResources(context);
        OclBuffer buf = createBuffer(context, sizeOf(cl_double, outShape.getSize()), CL_MEM_READ_WRITE);

        Kernel kernel = context.findKernel(CONV_2D);

        CommandQueue queue = context.getQueue();
        Device device = queue.getDevice();
        long totalExampleCount = calcExampleCount(input.shape, END_WITH__HEIGHT_WIDTH_CHANNEL);
        long outputChannels = filter.shape.at(-1);
        long totalOutputCoordCount = outputSize.calcArea() * outputChannels;

        long workGroupSizeMultiple = kernel.getPreferredWorkGroupSizeMultiple(queue);
        int singleOutputFilterVolume = calcVolumeForSingleOutput(filter.shape);
        int maxGroupSize = device.getMaxWorkGroupSize();

        int filterCalcPerWorkItem;
        int workItemsNeeded;
        if (singleOutputFilterVolume <= maxGroupSize) {
            filterCalcPerWorkItem = 1;
            workItemsNeeded = singleOutputFilterVolume;
        } else {
            filterCalcPerWorkItem = (singleOutputFilterVolume + maxGroupSize - 1) / maxGroupSize;
            long workItemsNeededWhenCalculatingMultipleInputs = (singleOutputFilterVolume + filterCalcPerWorkItem - 1) / filterCalcPerWorkItem;
            int workItemsNeededInPreferredMultiples = (int) ((workItemsNeededWhenCalculatingMultipleInputs + workGroupSizeMultiple - 1) / workGroupSizeMultiple * workGroupSizeMultiple);

            workItemsNeeded = workItemsNeededInPreferredMultiples;
        }

        kernel.createArgSetter(resources)
                .nextArg(input)
                .nextArg(input.shape.getSize())
                .nextArg(getInputSizes(input.shape))
                .nextArg(filter)
                .nextArg(filter.shape.getSize())
                .nextArg(filterOffsets)
                .nextArg(singleOutputFilterVolume)
                .nextArg(calcExampleBlockSize(input.shape))
                .nextArg(calcOutputBlockSizes(outputSize, filter.shape.at(-1)))
                .nextArg(calcFilterBlockSizes(filter.shape))
                .nextArgKeepRef(buf)
                .nextArg(calcOutputExampleBlockSize(outputSize, filter.shape.at(-1)))
                .nextArg(filterCalcPerWorkItem)
                .nextArg(workItemsNeeded)
                .nextArgLocalDoubles(workItemsNeeded)
        ;

        queue.enqueueKernel(kernel,
                new long[]{workItemsNeeded, totalOutputCoordCount, totalExampleCount},
                new long[]{workItemsNeeded, 1, 1},
                resources);

        return createNDArray(outShape, buf, resources);
    }

//    public static OclArray conv2d(Context context, OclArray input, OclArray filter,
//                                  int[] filterOffsets,
//                                  ConvSize outputSize) {
//        NDArray.validateConv2dShapes(input.shape, filter.shape);
//
//        OclShape outShape = evalOutputShape(context, input.shape, filter.shape, outputSize);
//
//        InProgressResources resources = new InProgressResources(context);
//        OclBuffer buf = createBuffer(context, sizeOf(cl_double, outShape.size), CL_MEM_READ_WRITE);
//
//        Kernel kernel = context.findKernel(CONV_2D);
//
////        blockSizes.sizes = new int[]{10, 20, 30};
//
//        CommandQueue queue = context.getQueue();
//        Device device = queue.getDevice();
//        long totalExampleCount = calcExampleCount(input.shape, END_WITH__HEIGHT_WIDTH_CHANNEL);
//        long outputChannels = filter.shape.at(-1);
//        long totalOutputCoordCount = outputSize.calcArea() * outputChannels;
//
//        long workGroupSizeMultiple = kernel.getPreferredWorkGroupSizeMultiple(queue);
//        int singleOutputFilterVolume = calcVolumeForSingleOutput(filter.shape);
//        int workGroupSize = min(calcWorkGroupSize(device, workGroupSizeMultiple), singleOutputFilterVolume);
//        int filterCalcPerWorkItem;
//        long workItemsNeeded;
//        if (singleOutputFilterVolume <= workGroupSize) {
//            filterCalcPerWorkItem = 1;
//            workItemsNeeded = singleOutputFilterVolume;
//        } else {
//            filterCalcPerWorkItem = (singleOutputFilterVolume + workGroupSize - 1) / workGroupSize;
////            System.out.println("filterCalcPerWorkItem = " + filterCalcPerWorkItem);
//            workItemsNeeded = (singleOutputFilterVolume + filterCalcPerWorkItem - 1) / filterCalcPerWorkItem;
//            long goodGroupSize = (workItemsNeeded + workGroupSizeMultiple - 1) / workGroupSizeMultiple * workGroupSizeMultiple;
//            workGroupSize = (int) goodGroupSize;
//            workItemsNeeded = workGroupSize;
//        }
//        // __global const double *inputs,
//        // __global const double *filter,
//        // int exampleBlockSize,
//        // int exampleWidth,
//        // __global const int *filterBlockSize,
//        // __global double *out,
//        // long outLen
//        kernel.createArgSetter(resources)
//                .nextArg(input)
//                .nextArg(input.shape.size)
//                .nextArg(getInputSizes(input.shape))
//                .nextArg(filter)
//                .nextArg(filter.shape.size)
//                .nextArg(filterOffsets)
//                .nextArg(singleOutputFilterVolume)
//                .nextArg(calcExampleBlockSize(input.shape))
//                .nextArg(calcOutputBlockSizes(outputSize, filter.shape.at(-1)))
//                .nextArg(calcFilterBlockSizes(filter.shape))
//                .nextArgKeepRef(buf)
//                .nextArg(calcOutputExampleBlockSize(outputSize, filter.shape.at(-1)))
//                .nextArg(filterCalcPerWorkItem)
//                .nextArg(workGroupSize)
//                .nextArgLocalDoubles(workGroupSize)
//        ;
//
//        queue.enqueueKernel(kernel,
//                new long[]{workItemsNeeded, totalOutputCoordCount, totalExampleCount},
//                new long[]{workGroupSize, 1, 1},
//                resources);
//
////        System.out.println("resources.opEvent.getValue() = " + resources.opEvent.getValue());
////        throwOnError(() -> OpenCL.INSTANCE.clWaitForEvents(1, OpenCL.PointerArray.pointers(resources.opEvent.getValue())));
//
//        return createNDArray(outShape, buf, resources);
//    }

    private static int calcOutputExampleBlockSize(ConvSize outputSize, int outChannels) {
        return (int) (outputSize.calcArea() * outChannels);
    }

    /**
     * @return [0]: height, [1]: width
     */
    private static int[] getInputSizes(Shape shape) {
        return new int[] {shape.at(-3), shape.at(-2), shape.at(-1)};
    }

    private static Shape evalOutputShape(Context context, Shape inputShape, Shape filterShape, ConvSize outputSize) {
        int dimCount = inputShape.getDimCount();
        int[] dims = new int[dimCount];
        for (int i = 0; i < dims.length; i++) {
            int size;
            if (i == dimCount-3) {
                size = outputSize.height;
            } else if(i == dimCount-2) {
                size = outputSize.width;
            } else if(i == dimCount-1) {
                size = filterShape.at(-1);
            } else {
                size = inputShape.at(i);
            }
            dims[i] = size;
        }

        return ProviderStore.shape(dims);
    }

    private static int[] calcOutputBlockSizes(ConvSize outputSize, int outChannels) {
        //#define OUT_BS_W_OUTCHAN        0
        //#define OUT_BS_OUTCHAN          1
        int[] r = new int[2];
        r[1] = outChannels;
        r[0] = r[1] * outputSize.width;
        return r;
    }

    private static int getExampleWidth(Shape shape) {
        return shape.at(-2);
    }

    private static int calcExampleBlockSize(Shape shape) {
        int channels = shape.at(-1);
        int width = shape.at(-2);
        int height = shape.at(-3);

        return channels * width * height;
    }

    private static int[] calcFilterBlockSizes(Shape shape) {
        //#define FLI_W_INCHAN            0
        //#define FLI_INCHAN              1
        int[] blocks = new int[2];
        blocks[1] = shape.at(-2);
        blocks[0] = blocks[1] * shape.at(-3);
        return blocks;
    }

    private static int calcWorkGroupSize(Device device, long workGroupSizeMultiple) {
        long maxSize = min(device.info.workItemSizes[0].longValue(), device.info.maxWorkGroupSize.longValue());

        return (int) (maxSize / workGroupSizeMultiple * workGroupSizeMultiple);
    }

    // Mul filter: height * width * inputChannels
    private static int calcVolumeForSingleOutput(Shape filterShape) {
        return filterShape.at(0) * filterShape.at(1) * filterShape.at(2);
    }

}
