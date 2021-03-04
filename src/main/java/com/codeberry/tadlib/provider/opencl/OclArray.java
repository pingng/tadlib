package com.codeberry.tadlib.provider.opencl;

import com.codeberry.tadlib.array.Comparison;
import com.codeberry.tadlib.array.NDArray;
import com.codeberry.tadlib.array.NDIntArray;
import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.array.util.FlatToMultiDimArrayConverter;
import com.codeberry.tadlib.array.util.SoftmaxUtils;
import com.codeberry.tadlib.provider.opencl.jna.TADMemory;
import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.opencl.buffer.BufferMemFlags;
import com.codeberry.tadlib.provider.opencl.context.Context;
import com.codeberry.tadlib.provider.opencl.ops.*;
import com.codeberry.tadlib.util.StringUtils;
import com.sun.jna.Pointer;

import java.util.*;
import java.util.function.Function;

import static com.codeberry.tadlib.memorymanagement.DisposalRegister.disposeAllExceptReturnedValue;
import static com.codeberry.tadlib.memorymanagement.DisposalRegister.registerForDisposal;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.*;
import static com.codeberry.tadlib.provider.opencl.OclDataType.*;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.ByteSize.sizeOf;
import static com.codeberry.tadlib.provider.opencl.OpenCL.PointerArray.usePointerArray;
import static com.codeberry.tadlib.provider.opencl.consts.ErrorCode.throwOnError;
import static com.codeberry.tadlib.provider.opencl.ops.Conv.ConvSize.Builder.convSizeBuilder;
import static java.lang.Math.*;


// for waiting: http://web.engr.oregonstate.edu/~mjb/cs575/Handouts/opencl.events.2pp.pdf
public class OclArray implements NDArray {
    public final OclBuffer buffer;
    public final Shape shape;

    final InProgressResources resources;

    private OclArray(Shape shape, OclBuffer buffer, InProgressResources resources) {
        this.shape = shape;
        this.buffer = buffer.lockOrCreateLockedView();
        this.resources = resources;
    }

    private OclArray(Context context, double v) {
        this(ProviderStore.shape(), createBuffer(context, sizeOf(cl_double, 1), BufferMemFlags.CL_MEM_READ_ONLY), new InProgressResources(context));

        TADMemory nativeBuf = resources.createDisposableMemory(cl_double.sizeOfElements(shape.getSize()));
        nativeBuf.setDouble(0, v);
        throwOnError(() -> OpenCL.INSTANCE.wrapperEnqueueWriteBuffer(
                context.getQueue(), this.buffer, false, new SizeT(0), new SizeT(nativeBuf.size()),
                nativeBuf, null, resources.opEvent));
    }

    private OclArray(Context context, Shape shape, double v) {
        this(shape, createBuffer(context, sizeOf(cl_double, shape.getSize()), BufferMemFlags.CL_MEM_READ_ONLY), new InProgressResources(context));

        this.buffer.oclFill(context.getQueue(), resources, v);
    }

    private OclArray(Context context, Shape shape, double[] v) {
        this(shape, createBuffer(context, sizeOf(cl_double, shape.getSize()), BufferMemFlags.CL_MEM_READ_ONLY), new InProgressResources(context));

        TADMemory nativeBuf = resources.createDisposableMemory(cl_double.sizeOfElements(shape.getSize()));
        nativeBuf.write(0, v, 0, v.length);
        throwOnError(() -> OpenCL.INSTANCE.wrapperEnqueueWriteBuffer(
                context.getQueue(), this.buffer, false, new SizeT(0), new SizeT(nativeBuf.size()),
                nativeBuf, null, resources.opEvent));
    }

    public static OclArray createNDArray(Shape shape, OclBuffer buf, InProgressResources resources) {
        return registerForDisposal(new OclArray(shape, buf, resources));
    }

    public static OclArray createNDArray(Context context, double v) {
        return registerForDisposal(new OclArray(context, v));
    }

    public static OclArray createNDArray(Context context, Shape shape, double[] data) {
        return registerForDisposal(new OclArray(context, shape, data));
    }

    public static NDArray createNDArray(Context context, Shape shape, double v) {
        return registerForDisposal(new OclArray(context, shape, v));
    }

    public Pointer getArgPointer() {
        return buffer.argPointer;
    }

    @Override
    public void waitForValueReady() {
        Pointer ev = resources.getKernelEvent();
        if (ev != null) {
            if (resources.isDisposed()) {
                throw new RuntimeException("Should not be disposed");
            }
            usePointerArray(pointers ->
                    throwOnError(() -> OpenCL.INSTANCE.clWaitForEvents(1, pointers),
                            resources::getContentStatus), ev);
        }
        resources.disposeDependencies();
    }

    @Override
    public NDArray negate() {
        return Simple.negate(buffer.context, this);
    }

    @Override
    public NDArray sqr() {
        return Simple.sqr(buffer.context, this);
    }

    @Override
    public NDArray sqrt() {
        return Simple.sqrt(buffer.context, this);
    }

    @Override
    public NDArray rot180(int yAxis, int xAxis) {
        return Simple.rot180(buffer.context, this, yAxis, xAxis);
    }

    @Override
    public NDArray pow(double val) {
        return Simple.pow(buffer.context, this, val);
    }

    @Override
    public NDArray add(NDArray other) {
        return Add.add(buffer.context, this, (OclArray) other);
    }

    @Override
    public NDArray add(double val) {
        return Add.add(buffer.context, this, val);
    }

    @Override
    public NDArray mul(NDArray other) {
        return Mul.mul(buffer.context, this, (OclArray) other);
    }

    @Override
    public NDArray div(NDArray other) {
        return Div.div(buffer.context, this, (OclArray) other);
    }

    @Override
    public NDArray mul(double val) {
        return Mul.mul(buffer.context, this, val);
    }

    @Override
    public NDArray div(double val) {
        return Div.div(buffer.context, this, val);
    }

    @Override
    public NDArray sum(Boolean[] dimsToCollapse, DimKeepRemove keepRemove) {
        return Sum.sum(buffer.context, this, dimsToCollapse, keepRemove);
    }

    @Override
    public MaxPool2dResult maxPool2d(int size) {
        return MaxPool.maxPool2d(buffer.context, this, size);
    }

    @Override
    public NDArray maxPool2dGrad(MaxPool2dResult result) {
        return MaxPool.maxPool2dGrad(buffer.context, this, result);
    }

    @Override
    public ReluResult relu(double leakyScale) {
        return Relu.relu(buffer.context, this, leakyScale);
    }

    @Override
    public NDArray softmax() {
        return Softmax.softmax(buffer.context, this);
    }

    @Override
    public NDArray softMaxCrossEntropyGrad(NDArray softmax, NDArray oneHotArray) {
        return SoftmaxUtils.calcSoftmaxCrossEntropyGradient(softmax, oneHotArray).mul(this);
    }

    /**
     * Reads data from OpenCL, then modify it and create result array.
     * <p>
     * TODO: Improve. Maybe create a mask that is multiplied with this array, to avoid reading from OpenCL.
     */
    @Override
    public DropOutResult dropOut(Random rnd, double dropoutKeep) {
        double[] vals = readFlatArray();

        double gradValue = 1.0 / dropoutKeep;
        double[] gradMaskData = new double[vals.length];
        for (int i = 0; i < vals.length; i++) {
            if (rnd.nextDouble() >= dropoutKeep) {
                vals[i] = 0;
            } else {
                vals[i] /= dropoutKeep;
                gradMaskData[i] = gradValue;
            }
        }

        NDArray output = ProviderStore.array(vals, shape);
        return new DropOutResult() {
            @Override
            public NDArray getOutput() {
                return output;
            }

            @Override
            public NDArray createMask() {
                return ProviderStore.array(gradMaskData, shape);
            }
        };
    }

    private double[] readFlatArray() {
        int size = toIntExact(shape.getSize());
        double[] vals = readToNative(nativeBuf -> {
            double[] dbl = new double[size];
            nativeBuf.read(0, dbl, 0, size);
            return dbl;
        });
        return vals;
    }

    @Override
    public NDArray withUpdates(List<ValueUpdate> updates) {
        return Update.update(buffer.context, this, updates);
    }

    @Override
    public NDArray conv2d(NDArray filter, int offsetY, int offsetX) {
        int filterHeight = filter.getShape().at(0);
        int filterWidth = filter.getShape().at(1);

        int outHeight = shape.at(-3);
        int outWidth = shape.at(-2);

        int diffY = getShape().at(-3) - outHeight;
        int diffX = getShape().at(-2) - outWidth;

        return conv2d(filter,
                offsetY - (filterHeight - 1) / 2 + diffY / 2,
                offsetX - (filterWidth - 1) / 2 + diffX / 2,
                outHeight, outWidth);
    }

    @Override
    public NDArray conv2d(NDArray filter, int offsetY, int offsetX, int outHeight, int outWidth) {
        return Conv.conv2d(buffer.context, this, (OclArray) filter,
                new int[]{
                        offsetY,
                        offsetX},
                convSizeBuilder()
                        .height(outHeight)
                        .width(outWidth)
                        .build());
    }

//    @Override
//    public NDArray calcConv2dFilterGradient(NDArray input, NDArray filter) {
////        System.out.println("============");
//        Shape filterShape = filter.getShape();
////        System.out.println("Filter.shape: "+ filterShape);
//        testNan(this);
//        NDArray gradTransposed = this.transpose(1, 2, 0, 3);
//        testNan(gradTransposed);
////        print("== Grad", this, gradTransposed, 3,3);
//
//        testNan(input);
//        NDArray inputTransposed = input.transpose(3, 1, 2, 0);
//        testNan(inputTransposed);
////        print("== Input", input, inputTransposed, 3,3);
//
//        int offsetY = -(filter.getShape().at(0)-1)/2;
//        int offsetX = -(filter.getShape().at(1)-1)/2;
//        NDArray tmpOut = ((OclArray)inputTransposed).conv2dTry(gradTransposed, offsetY, offsetX,
//                filterShape.at(0),
//                filterShape.at(1));
//        testNan(tmpOut);
////        System.out.println("tmpOut.shape: "+tmpOut.getShape());
//
//        NDArray out = tmpOut.transpose(1, 2, 0, 3);
//        testNan(out);
//        //NDArray outRot180 = out.rot180(0, 1);
////        System.out.println("out.shape: "+out.getShape());
////        System.out.println("out: " + toJson(out.toDoubles(), COMPACT));
//
//        //return outRot180;
//        return out;
//        //Inp: (batch, Hi, Wi, Ci)
//        //
//        //Flt: (Hf, Wf, Ci, Co)
//        //
//        //Out: (batch, Ho, Wo, Co)
//        //
//        //
//        //Grd transpose: (Co, Ho, Wo, batch)
//        //
//        //Inp transpose: (Hi, Wi, batch, Ci)
//        //
//        //Res:           (Co, Hf, Wf, Ci) -> (Hf, Wf, Ci, Co)
//    }

    @Override
    public NDArray calcConv2dFilterGradient(NDArray input, NDArray filter) {
        return disposeAllExceptReturnedValue(() -> {
            Shape filterShape = filter.getShape();
            NDArray gradTransposed = this.transpose(1, 2, 0, 3);
            NDArray inputTransposed = input.transpose(3, 1, 2, 0);

            int offsetY = -(filter.getShape().at(0) - 1) / 2;
            int offsetX = -(filter.getShape().at(1) - 1) / 2;
            NDArray tmpOut = inputTransposed.conv2d(gradTransposed, offsetY, offsetX,
                    filterShape.at(0),
                    filterShape.at(1));

            return tmpOut.transpose(1, 2, 0, 3);
        });
    }

    public void dispose() {
        this.buffer.dispose();
        this.resources.disposeDependencies();
    }

    @Override
    public NDArray clip(Double min, Double max) {
        return Clip.clip(buffer.context, this, min, max);
    }

    @Override
    public NDArray log() {
        return Simple.log(buffer.context, this);
    }

    @Override
    public NDIntArray argmax(int axis) {
        return ArgMax.argMax(buffer.context, this, axis);
    }

    @Override
    public NDArray getAtIndicesOnAxis(NDIntArray indices, int axis) {
        return GetAtIndicesOnAxis.getAtIndicesOnAxis(buffer.context, this, (OclIntArray) indices, axis);
    }

    @Override
    public NDArray withUpdateAtIndicesOnAxis(NDIntArray indices, int axis, NDArray change) {
        return UpdateAtIndicesOnAxis.updateAtIndicesOnAxis(buffer.context, this, (OclIntArray) indices, axis, (OclArray) change);
    }

    @Override
    public NDArray diag() {
        return Diagonal.diag(buffer.context, this);
    }

    @Override
    public NDArray concat(NDArray[] appendees, int axis) {
        OclArray[] copy = new OclArray[appendees.length + 1];
        System.arraycopy(appendees, 0, copy, 1, appendees.length);
        copy[0] = this;
        return Concat.concat(buffer.context, copy, getShape().wrapNegIndex(axis));
    }

    @Override
    public List<NDArray> split(int axis, int[] axisLens) {
        return Split.split(buffer.context, this, getShape().wrapNegIndex(axis), axisLens);
    }

    @Override
    public NDArray matmul(NDArray b) {
        return MatMul.matmul(buffer.context, this, (OclArray) b);
    }

    @Override
    public NDArray transpose(int... axes) {
        return Transpose.transpose(buffer.context, this, axes);
    }

    @Override
    public NDArray compare(NDIntArray other, Comparison comparison, double trueValue, double falseValue) {
        return Compare.compare(buffer.context, this, (OclIntArray) other, comparison, trueValue, falseValue);
    }

    @Override
    public NDArray compare(NDArray other, Comparison comparison, double trueValue, double falseValue) {
        return Compare.compare(buffer.context, this, (OclArray) other, comparison, trueValue, falseValue);
    }

    @Override
    public Shape getShape() {
        return shape;
    }

    @Override
    public NDArray reshape(int... dims) {
        return createNDArray(this.shape.reshape(dims), buffer, resources);
    }

    @Override
    public Object toDoubles() {
        return readToNative(nativeBuf ->
                FlatToMultiDimArrayConverter.toDoubles(this.shape,
                        i -> nativeBuf.getDouble(i * cl_double.byteSize)));
    }

    private <E> E readToNative(Function<TADMemory, E> mapper) {
        try(TADMemory nativeBuf = new TADMemory(cl_double.sizeOfElements(shape.getSize()))) {
            Pointer kernelEvent = resources.getKernelEvent();

            usePointerArray(events ->
                    throwOnError(() -> OpenCL.INSTANCE.wrapperEnqueueReadBuffer(buffer.context.getQueue(),
                            buffer, true,
                            new SizeT(0),
                            new SizeT(nativeBuf.size()), nativeBuf, events,
                            null)), kernelEvent);

            resources.disposeDependencies();

            return mapper.apply(nativeBuf);
        }
    }

    @Override
    public NDArray subArray(int fromBatchIndex, int fromOffset, int endBatchIndex, int toOffset) {

        InProgressResources res = new InProgressResources(buffer.context);
        OclBuffer buf = this.buffer.oclDoubleSubBuffer(res, fromOffset, toOffset);
        int[] dims = this.shape.toDimArray();
        dims[0] = endBatchIndex - fromBatchIndex;

        return createNDArray(ProviderStore.shape(dims), buf, res);
    }

    @Override
    public NDArray normalOrderedCopy() {
        return this;
    }

    @Override
    public double[] getInternalData() {
        return readFlatArray();
    }

    @Override
    public double dataAt(int... indices) {
        int index = getShape().calcDataIndex(indices);
        return buffer.oclReadDouble(resources, index);
    }

    /**
     * Meant for tracking & debugging buffers that was unintentionally released.
     */
    public void setBufferDisposeCallback(Runnable r) {
        this.buffer.setDisposeCallback(r);
    }

    @Override
    public String toString() {
        return StringUtils.toString(this);
    }

}
