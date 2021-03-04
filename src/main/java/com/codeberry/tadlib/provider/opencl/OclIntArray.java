package com.codeberry.tadlib.provider.opencl;

import com.codeberry.tadlib.array.Comparison;
import com.codeberry.tadlib.array.NDArray;
import com.codeberry.tadlib.array.NDIntArray;
import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.array.util.FlatToMultiDimArrayConverter;
import com.codeberry.tadlib.provider.opencl.jna.TADMemory;
import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.opencl.buffer.BufferMemFlags;
import com.codeberry.tadlib.provider.opencl.context.Context;
import com.codeberry.tadlib.provider.opencl.ops.Compare;
import com.sun.jna.Pointer;

import java.util.function.Function;

import static com.codeberry.tadlib.memorymanagement.DisposalRegister.registerForDisposal;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.ByteSize.sizeOf;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.createBuffer;
import static com.codeberry.tadlib.provider.opencl.OclDataType.cl_int;
import static com.codeberry.tadlib.provider.opencl.OpenCL.PointerArray.usePointerArray;
import static com.codeberry.tadlib.provider.opencl.consts.ErrorCode.throwOnError;

public class OclIntArray implements NDIntArray {

    private final Shape shape;
    final OclBuffer buffer;

    final InProgressResources resources;

    private OclIntArray(Shape shape, OclBuffer buffer, InProgressResources resources) {
        this.shape = shape;
        this.buffer = buffer.lockOrCreateLockedView();
        this.resources = resources;
    }

    private OclIntArray(Context context, int v) {
        this(ProviderStore.shape(), createBuffer(context, sizeOf(cl_int, 1), BufferMemFlags.CL_MEM_READ_ONLY), new InProgressResources(context));

        TADMemory nativeBuf = resources.createDisposableMemory(cl_int.sizeOfElements(shape.getSize()));
        nativeBuf.setInt(0, v);
        throwOnError(() -> OpenCL.INSTANCE.wrapperEnqueueWriteBuffer(
                context.getQueue(), this.buffer, false, new SizeT(0), new SizeT(nativeBuf.size()),
                nativeBuf, null, resources.opEvent));
    }

    public OclIntArray(Context context, Shape shape, int[] v) {
        this(shape, createBuffer(context, sizeOf(cl_int, shape.getSize()), BufferMemFlags.CL_MEM_READ_ONLY), new InProgressResources(context));

        TADMemory nativeBuf = resources.createDisposableMemory(cl_int.sizeOfElements(shape.getSize()));
        nativeBuf.write(0, v, 0, v.length);
        throwOnError(() -> OpenCL.INSTANCE.wrapperEnqueueWriteBuffer(
                context.getQueue(), this.buffer, false, new SizeT(0), new SizeT(nativeBuf.size()),
                nativeBuf, null, resources.opEvent));

    }

    private OclIntArray(Context context, Shape shape, int v) {
        this(shape, createBuffer(context, sizeOf(cl_int, shape.getSize()), BufferMemFlags.CL_MEM_READ_ONLY), new InProgressResources(context));

        this.buffer.oclFill(context.getQueue(), resources, v);
    }

    public static NDIntArray createNDArray(Context context, Shape shape, int[] data) {
        return registerForDisposal(new OclIntArray(context, shape, data));
    }

    public static OclIntArray createNDArray(Shape shape, OclBuffer buf, InProgressResources resources) {
        return registerForDisposal(new OclIntArray(shape, buf, resources));
    }

    public static OclIntArray createNDArray(Context context, int v) {
        return registerForDisposal(new OclIntArray(context, v));
    }

    public static OclIntArray createNDArray(Context context, Shape shape, int v) {
        return registerForDisposal(new OclIntArray(context, shape, v));
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

    public Pointer getArgPointer() {
        return buffer.argPointer;
    }

    public void dispose() {
        this.buffer.dispose();
        this.resources.disposeDependencies();
    }

    private <E> E readToNative(Function<TADMemory, E> mapper) {
        try (TADMemory nativeBuf = new TADMemory(cl_int.sizeOfElements(shape.getSize()))) {

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
    public Object toInts() {
        return readToNative(nativeBuf -> FlatToMultiDimArrayConverter.toInts(this.shape,
                i -> nativeBuf.getInt(i * cl_int.byteSize)));
    }

    @Override
    public Shape getShape() {
        return shape;
    }

    @Override
    public int dataAt(int... indices) {
        return 0;
    }

    @Override
    public NDIntArray reshape(int... dims) {
        return createNDArray(this.shape.reshape(dims), buffer, resources);
    }

    @Override
    public NDArray compare(NDArray other, Comparison comparison, double trueValue, double falseValue) {
        return Compare.compare(buffer.context, this, (OclArray) other, comparison, trueValue, falseValue);
    }

    @Override
    public NDIntArray compare(NDIntArray other, Comparison comparison, int trueValue, int falseValue) {
        return Compare.compare(buffer.context, this, (OclIntArray) other, comparison, trueValue, falseValue);
    }
}
