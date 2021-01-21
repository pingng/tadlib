package com.codeberry.tadlib.provider.opencl;

import com.codeberry.tadlib.provider.opencl.jna.TADByReference;
import com.codeberry.tadlib.provider.opencl.jna.TADDoubleByReference;
import com.codeberry.tadlib.provider.opencl.jna.TADIntByReference;
import com.codeberry.tadlib.provider.opencl.jna.TADMemory;
import com.codeberry.tadlib.memorymanagement.AbstractDisposer;
import com.codeberry.tadlib.memorymanagement.DisposalRegister;
import com.codeberry.tadlib.provider.opencl.buffer.BufferMemFlags;
import com.codeberry.tadlib.provider.opencl.consts.ErrorCode;
import com.codeberry.tadlib.provider.opencl.context.Context;
import com.codeberry.tadlib.provider.opencl.queue.CommandQueue;
import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Pointer;

import static com.codeberry.tadlib.provider.opencl.OclDataType.cl_double;
import static com.codeberry.tadlib.provider.opencl.OclDataType.cl_int;
import static com.codeberry.tadlib.provider.opencl.buffer.BufferMemFlags.CL_MEM_READ_ONLY;
import static com.codeberry.tadlib.provider.opencl.consts.ErrorCode.throwOnError;
import static java.lang.Math.min;

public class OclBuffer extends Pointer implements DisposalRegister.Disposable {

    final Context context;
    public final TADMemory argPointer;
    private final long byteCount;
    private boolean locked;

    private final Disposer disposer;

    private OclBuffer(Pointer rawOclPointer, Context context, long byteCount) {
        super(Pointer.nativeValue(rawOclPointer));

        this.context = context;
        this.argPointer = createMemoryPointerTo(rawOclPointer);
        this.byteCount = byteCount;

        this.disposer = new Disposer(Pointer.nativeValue(rawOclPointer));
    }

    private OclBuffer(OclBuffer original) {
        super(Pointer.nativeValue(original));

        this.context = original.context;
        this.argPointer = createMemoryPointerTo(original);
        this.byteCount = original.byteCount;

        this.disposer = new Disposer(Pointer.nativeValue(original), original.disposer);

        // Increase the reference count in opencl
        // (ref count is released by disposer)
        throwOnError(() -> OpenCL.INSTANCE.wrapperRetainMemObject(this));
    }

    public void setDisposeCallback(Runnable r) {
        disposer.setCallback(r);
    }

    public long getByteCount() {
        return byteCount;
    }

    public ErrorCode canBeRead() {
        Memory nativeBuf = new Memory(this.byteCount);

        int retCode = OpenCL.INSTANCE.wrapperEnqueueReadBuffer(context.getQueue(),
                this, true,
                new SizeT(0),
                new SizeT(nativeBuf.size()), nativeBuf, null,
                null);

        return ErrorCode.errorOf(retCode);
    }


    private static class Disposer extends AbstractDisposer {
        private final long oclBufferPointerAddr;

        Disposer(long oclBufferPointerAddr) {
            super();

            this.oclBufferPointerAddr = oclBufferPointerAddr;
        }

        Disposer(long oclBufferPointerAddr, Disposer parent) {
            super(parent);

            this.oclBufferPointerAddr = oclBufferPointerAddr;
        }

        @Override
        protected void releaseResource() {
            OpenCL.INSTANCE.wrapperReleaseMemObject(Pointer.createConstant(oclBufferPointerAddr));
        }

        @Override
        protected long getResourceId() {
            return oclBufferPointerAddr;
        }
    }

    @Override
    public void prepareDependenciesForDisposal() {
        // do nothing
    }

    public synchronized void dispose() {
        disposer.release();
        Pointer.nativeValue(this, 0);
        argPointer.dispose();
    }

    private synchronized void enableLock() {
        locked = true;
    }

    public synchronized OclBuffer lockOrCreateLockedView() {
        if (!locked) {
            enableLock();
            return this;
        }
        OclBuffer view = createView();
        view.enableLock();
        return view;
    }

    public synchronized OclBuffer createView() {
        return disposer.runBlockingConcurrentRelease(() -> OclBuffer.createBuffer(this));
    }

    private static TADMemory createMemoryPointerTo(Pointer pointer) {
        TADMemory mem = new TADMemory(Native.POINTER_SIZE);
        mem.setPointer(0, pointer);
        return mem;
    }

    private static OclBuffer createBuffer(OclBuffer src) {
        return registerForCleanupAtGC(new OclBuffer(src));
    }

    public static OclBuffer createBuffer(Context context, ByteSize size, BufferMemFlags flags) {
        OpenCL inst = OpenCL.INSTANCE;

        long byteCount = size.getByteCount();
        Pointer rawOclPointer = throwOnError(errCode ->
                inst.wrapperCreateBuffer(context, flags.bits, new SizeT(byteCount), null, errCode));

        return registerForCleanupAtGC(new OclBuffer(rawOclPointer, context, byteCount));
    }

    private static OclBuffer registerForCleanupAtGC(OclBuffer buffer) {
        DisposalRegister.registerDisposer(buffer, buffer.disposer);

        return buffer;
    }

    public void oclFill(CommandQueue queue, InProgressResources resources, double value) {
        fill(queue, resources, new TADDoubleByReference(value), cl_double);
    }

    public void oclFill(CommandQueue queue, InProgressResources resources, int value) {
        fill(queue, resources, new TADIntByReference(value), cl_int);
    }

    private void fill(CommandQueue queue, InProgressResources resources, TADByReference valueRef, OclDataType dataType) {
        resources.useDependencyEvents(events -> {
            OclEventByReference event = new OclEventByReference();

            throwOnError(() -> OpenCL.INSTANCE.wrapperEnqueueFillBuffer(queue,
                    this,
                    valueRef.getPointer(),
                    new SizeT(dataType.byteSize),
                    new SizeT(0),
                    new SizeT(this.byteCount),
                    events,
                    event));

            resources.registerDisposableByRef(valueRef);
            resources.registerDisposableByRef(event);
        });
    }

    public void oclCopy(CommandQueue queue, InProgressResources resources, OclBuffer src, long srcOffset) {
        resources.useDependencyEvents(events -> {
            OclEventByReference event = new OclEventByReference();

            long minSize = min(this.byteCount, src.byteCount - srcOffset);

            throwOnError(() -> OpenCL.INSTANCE.wrapperEnqueueCopyBuffer(queue,
                    src,
                    this,
                    new SizeT(srcOffset),
                    new SizeT(0),
                    new SizeT(minSize),
                    events,
                    event));

            resources.registerDisposableByRef(event);
        });
    }

    public double oclReadDouble(InProgressResources resources, int index) {
        TADMemory nativeBuf = new TADMemory(cl_double.sizeOfElements(1));

        resources.useDependencyEvents(events ->
                throwOnError(() -> OpenCL.INSTANCE.wrapperEnqueueReadBuffer(context.getQueue(),
                        this, true,
                        new SizeT(cl_double.sizeOfElements(index)),
                        new SizeT(nativeBuf.size()), nativeBuf, events,
                        null)));

        double value = nativeBuf.getDouble(0);
        nativeBuf.dispose();

        return value;
    }

    public OclBuffer oclDoubleSubBuffer(InProgressResources res, int fromOffset, int toOffset) {
        OclBuffer copy = createBuffer(context, ByteSize.sizeOf(cl_double, toOffset - fromOffset), CL_MEM_READ_ONLY);
        copy.oclCopy(context.getQueue(), res, this, cl_double.sizeOfElements(fromOffset));
        return copy;
    }

    public static class ByteSize {
        private final OclDataType type;
        private final long numberOf;

        public ByteSize(OclDataType type, long numberOf) {
            this.type = type;
            this.numberOf = numberOf;
        }

        public static ByteSize sizeOf(OclDataType type, long numberOf) {
            return new ByteSize(type, numberOf);
        }

        public long getByteCount() {
            return type.byteSize * numberOf;
        }
    }
}
