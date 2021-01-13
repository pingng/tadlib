package com.codeberry.tadlib.provider.opencl;

import com.codeberry.tadlib.memorymanagement.AbstractDisposer;
import com.codeberry.tadlib.memorymanagement.DisposalRegister;
import com.codeberry.tadlib.provider.opencl.buffer.BufferMemFlags;
import com.codeberry.tadlib.provider.opencl.consts.ErrorCode;
import com.codeberry.tadlib.provider.opencl.context.Context;
import com.codeberry.tadlib.provider.opencl.queue.CommandQueue;
import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.DoubleByReference;

import static com.codeberry.tadlib.provider.opencl.OclDataType.cl_double;
import static com.codeberry.tadlib.provider.opencl.buffer.BufferMemFlags.CL_MEM_READ_ONLY;
import static com.codeberry.tadlib.provider.opencl.consts.ErrorCode.throwOnError;
import static java.lang.Math.min;

public class OclBuffer extends Pointer implements DisposalRegister.Disposable {

    final Context context;
    public final Pointer argPointer;
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
        argPointer.setPointer(0, null);
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

    private static Memory createMemoryPointerTo(Pointer pointer) {
        Memory mem = new Memory(Native.POINTER_SIZE);
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

    public void oclFill(CommandQueue queue, OclArray.InProgressResources resources, double value) {
        OpenCL.PointerArray events = resources.getDependencyEvents();
        OclEventByReference event = new OclEventByReference();

        DoubleByReference valueRef = new DoubleByReference(value);
        throwOnError(() -> OpenCL.INSTANCE.wrapperEnqueueFillBuffer(queue,
                this,
                valueRef.getPointer(),
                new SizeT(OclDataType.cl_double.byteSize),
                new SizeT(0),
                new SizeT(this.byteCount),
                events,
                event));

        resources.registerDependencyEvent(event);
    }

    public void oclCopy(CommandQueue queue, OclArray.InProgressResources resources, OclBuffer src, long srcOffset) {
        OpenCL.PointerArray events = resources.getDependencyEvents();
        OclEventByReference event = new OclEventByReference();

        long minSize = min(this.byteCount, src.byteCount);

        throwOnError(() -> OpenCL.INSTANCE.wrapperEnqueueCopyBuffer(queue,
                src,
                this,
                new SizeT(srcOffset),
                new SizeT(0),
                new SizeT(minSize),
                events,
                event));

        resources.registerDependencyEvent(event);
    }

    public double oclReadDouble(OclArray.InProgressResources resources, int index) {
        Memory nativeBuf = new Memory(cl_double.sizeOfElements(1));

        OpenCL.PointerArray events = resources.getDependencyEvents();

        throwOnError(() -> OpenCL.INSTANCE.wrapperEnqueueReadBuffer(context.getQueue(),
                this, true,
                new SizeT(cl_double.sizeOfElements(index)),
                new SizeT(nativeBuf.size()), nativeBuf, events,
                null));

        return nativeBuf.getDouble(0);
    }

    public OclBuffer oclDoubleSubBuffer(OclArray.InProgressResources res, int fromOffset, int toOffset) {
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
