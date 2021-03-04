package com.codeberry.tadlib.provider.opencl;

import com.codeberry.tadlib.provider.opencl.jna.*;
import com.codeberry.tadlib.provider.opencl.buffer.BufferMemFlags;
import com.codeberry.tadlib.provider.opencl.context.Context;
import com.sun.jna.Native;
import com.sun.jna.Pointer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Stream;

import static com.codeberry.tadlib.provider.opencl.OclBuffer.ByteSize.sizeOf;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.createBuffer;
import static com.codeberry.tadlib.provider.opencl.OclDataType.*;
import static com.codeberry.tadlib.provider.opencl.OpenCL.PointerArray.mapPointerArray;
import static com.codeberry.tadlib.provider.opencl.consts.ErrorCode.throwOnError;
import static java.lang.Math.toIntExact;
import static java.util.stream.Collectors.toList;

/**
 * Holds on to objects until disposed/finalized.
 * <p>
 * For preventing GC of source objects.
 */
public class InProgressResources {
    private final Context context;

    public volatile OclEventByReference opEvent;
    private List<TADMemory> disposableMemory = new ArrayList<>();
    private List<TADByReference> disposableByRefs = new ArrayList<>();
    private List<OclBuffer> disposableBuffers = new ArrayList<>();
    private List<OclBuffer> strongReferredBuffers = new ArrayList<>();
    private List<OclArray> strongReferredNDArrays = new ArrayList<>();
    private List<OclIntArray> strongReferredNDIntArrays = new ArrayList<>();

    public InProgressResources(Context context) {
        this.context = context;

        opEvent = new OclEventByReference();
        disposableByRefs.add(opEvent);
    }

    public void registerDependency(OclArray dependantArray) {
        if (dependantArray != null) {
            this.strongReferredNDArrays.add(dependantArray);
        }
    }

    public void registerDependency(OclIntArray dependantArray) {
        if (dependantArray != null) {
            this.strongReferredNDIntArrays.add(dependantArray);
        }
    }

    public void registerDisposableBuffer(OclBuffer dependantArray) {
        if (dependantArray != null) {
            this.disposableBuffers.add(dependantArray);
        }
    }

    public void registerReferredBuffer(OclBuffer keepRefToBuffer) {
        if (keepRefToBuffer != null) {
            this.strongReferredBuffers.add(keepRefToBuffer);
        }
    }

    public void useDependencyEvents(Consumer<OpenCL.PointerArray> consumer) {
        Pointer[] events = toEventPointers(() -> Stream.concat(
                strongReferredNDArrays.stream().map(a -> a.resources),
                strongReferredNDIntArrays.stream().map(a -> a.resources)));

        OpenCL.PointerArray.usePointerArray(consumer, events);
    }

    private static Pointer[] toEventPointers(Supplier<Stream<InProgressResources>> resourcesSupplier) {
        Pointer[] evPointers = writeKernelEvents(resourcesSupplier.get());
        if (evPointers.length > 0) {
            return evPointers;
        }
        return null;
    }

    private static Pointer[] writeKernelEvents(Stream<InProgressResources> resources) {
        return resources
                .map(InProgressResources::getKernelEvent)
                .filter(Objects::nonNull)
                .toArray(Pointer[]::new);
    }

    public Pointer argLong(long v) {
        TADLongByReference ref = new TADLongByReference(v);
        disposableByRefs.add(ref);

        return ref.getPointer();
    }

    public Pointer argInt(int v) {
        TADIntByReference ref = new TADIntByReference(v);
        disposableByRefs.add(ref);

        return ref.getPointer();
    }

    public Pointer argDouble(double v) {
        TADDoubleByReference ref = new TADDoubleByReference(v);
        disposableByRefs.add(ref);

        return ref.getPointer();
    }

    public Pointer argBoolean(boolean v) {
        TADIntByReference ref = new TADIntByReference(v ? 1 : 0);
        disposableByRefs.add(ref);

        return ref.getPointer();
    }

    public synchronized boolean isDisposed() {
        return opEvent == null;
    }

    public synchronized void disposeDependencies() {
        for (OclArray ndArr : strongReferredNDArrays) {
            ndArr.resources.disposeDependencies();
        }
        for (OclIntArray ndArr : strongReferredNDIntArrays) {
            ndArr.resources.disposeDependencies();
        }
        for (TADByReference event : disposableByRefs) {
            event.dispose();
        }
        for (OclBuffer buf : disposableBuffers) {
            buf.dispose();
        }
        for (TADMemory mp : disposableMemory) {
            mp.dispose();
        }

        strongReferredNDArrays.clear();
        strongReferredNDIntArrays.clear();
        strongReferredBuffers.clear();
        disposableBuffers.clear();
        disposableByRefs.clear();
        disposableMemory.clear();

        OclEventByReference ev = this.opEvent;
        if (ev != null) {
            ev.dispose();
            this.opEvent = null;
        }
    }

    public Pointer registerReadOnlyArg(int[] v) {
        if (v != null && v.length > 0) {
            TADMemory memory = createMemory(v);
            OclBuffer buf = createBuffer(context, sizeOf(cl_int, v.length), BufferMemFlags.CL_MEM_READ_ONLY);
            OclEventByReference bufferWriteEvt = new OclEventByReference();
            throwOnError(() -> OpenCL.INSTANCE.wrapperEnqueueWriteBuffer(
                    context.getQueue(), buf, false, new SizeT(0), new SizeT(memory.size()),
                    memory, null, bufferWriteEvt));

            TADMemory pointerToPointer = new TADMemory(Native.POINTER_SIZE);
            pointerToPointer.setPointer(0, buf);

            disposableMemory.add(memory);
            disposableMemory.add(pointerToPointer);
            disposableByRefs.add(bufferWriteEvt);

            registerDisposableBuffer(buf);

            return pointerToPointer;
        }
        return Pointer.NULL;
    }

    public Pointer registerReadOnlyArg(long[] v) {
        if (v != null) {
            TADMemory memory = createMemory(v);
            OclBuffer buf = createBuffer(context, sizeOf(cl_long, v.length), BufferMemFlags.CL_MEM_READ_ONLY);
            OclEventByReference bufferWriteEvt = new OclEventByReference();
            throwOnError(() -> OpenCL.INSTANCE.wrapperEnqueueWriteBuffer(
                    context.getQueue(), buf, false, new SizeT(0), new SizeT(memory.size()),
                    memory, null, bufferWriteEvt));

            TADMemory pointerToPointer = new TADMemory(Native.POINTER_SIZE);
            pointerToPointer.setPointer(0, buf);

            disposableMemory.add(memory);
            disposableMemory.add(pointerToPointer);
            disposableByRefs.add(bufferWriteEvt);

            registerDisposableBuffer(buf);

            return pointerToPointer;
        }
        return Pointer.NULL;
    }

    public Pointer registerReadOnlyArg(double[] v) {
        if (v != null) {
            TADMemory memory = createMemory(v);
            OclBuffer buf = createBuffer(context, sizeOf(cl_double, v.length), BufferMemFlags.CL_MEM_READ_ONLY);
            OclEventByReference bufferWriteEvt = new OclEventByReference();
            throwOnError(() -> OpenCL.INSTANCE.wrapperEnqueueWriteBuffer(
                    context.getQueue(), buf, false, new SizeT(0), new SizeT(memory.size()),
                    memory, null, bufferWriteEvt));

            TADMemory pointerToPointer = new TADMemory(Native.POINTER_SIZE);
            pointerToPointer.setPointer(0, buf);

            disposableMemory.add(memory);
            disposableMemory.add(pointerToPointer);
            disposableByRefs.add(bufferWriteEvt);

            registerDisposableBuffer(buf);

            return pointerToPointer;
        }
        return Pointer.NULL;
    }

    private static TADMemory createMemory(int[] intArr) {
        TADMemory dimBlockSizesLeftPointer = new TADMemory(cl_int.sizeOfElements(intArr.length));
        dimBlockSizesLeftPointer.write(0, intArr, 0, intArr.length);
        return dimBlockSizesLeftPointer;
    }

    private static TADMemory createMemory(long[] longArr) {
        TADMemory dimBlockSizesLeftPointer = new TADMemory(cl_long.sizeOfElements(longArr.length));
        dimBlockSizesLeftPointer.write(0, longArr, 0, longArr.length);
        return dimBlockSizesLeftPointer;
    }

    private static TADMemory createMemory(double[] doubleArr) {
        TADMemory dimBlockSizesLeftPointer = new TADMemory(cl_double.sizeOfElements(doubleArr.length));
        dimBlockSizesLeftPointer.write(0, doubleArr, 0, doubleArr.length);
        return dimBlockSizesLeftPointer;
    }

    public <R extends TADByReference> R registerDisposableByRef(R ref) {
        Objects.requireNonNull(ref, "Cannot add NULL dependency ref");

        disposableByRefs.add(ref);

        return ref;
    }

    public TADMemory createDisposableMemory(long sizeOfElements) {
        TADMemory r = new TADMemory(sizeOfElements);
        disposableMemory.add(r);
        return r;
    }

    String getContentStatus() {
        StringBuilder sb = new StringBuilder();
        tryReadBuffers("StrongRefBuf", sb, strongReferredBuffers);
        tryReadBuffers("DisposableRefBuf", sb, disposableBuffers);
        tryReadBuffers("NDArray.buffer", sb, extractBuffers(strongReferredNDArrays, arr -> arr.buffer));
        tryReadBuffers("NDIntArray.buffer", sb, extractBuffers(strongReferredNDIntArrays, arr -> arr.buffer));
        for (int i = 0; i < disposableMemory.size(); i++) {
            TADMemory m = disposableMemory.get(i);
            sb.append("Memory[").append(i).append("/").append(disposableMemory.size()).append("]:").append(m.size()).append(":readOk=");
            try {
                synchronized (m) {
                    if (m.isDisposed()) {
                        sb.append("(disposed)");
                    } else {
                        byte[] tmp = new byte[toIntExact(m.size())];
                        m.read(0, tmp, 0, tmp.length);
                        sb.append("true");
                    }
                }
            } catch (Exception e) {
                sb.append("false").append(e.toString());
            }
            sb.append("\n");
        }
        for (int i = 0; i < disposableByRefs.size(); i++) {
            TADByReference ref = disposableByRefs.get(i);
            sb.append("Reference[").append(i).append("/").append(disposableByRefs.size()).append("]:").append(ref).append("\n");
        }

        useDependencyEvents(events -> {
            if (events != null) {
                for (int i = 0; i < events.length(); i++) {
                    sb.append("Event[").append(i).append("/").append(events.length()).append("]:waitRet=")
                            .append((int) mapPointerArray(pArr -> OpenCL.INSTANCE.clWaitForEvents(1, pArr),
                                            events.getPointer((long) i * Native.POINTER_SIZE)))
                            .append("\n");
                }
            }
        });

        return sb.toString();
    }

    private static <E> List<OclBuffer> extractBuffers(List<E> arrays, Function<E, OclBuffer> oclArrayOclBufferFunction) {
        return arrays.stream()
                .map(oclArrayOclBufferFunction)
                .collect(toList());
    }

    private static void tryReadBuffers(final String prefix, StringBuilder sb, List<OclBuffer> buffers) {
        for (int i = 0; i < buffers.size(); i++) {
            OclBuffer buf = buffers.get(i);

            sb.append(prefix).append("[").append(i).append("/").append(buffers.size()).append("]" +
                    "(").append(buf.getByteCount()).append("):")
                    .append(buf.canBeRead()).append("\n");
        }
    }

    Pointer getKernelEvent() {
        if (opEvent != null) {
            return opEvent.getValue();
        }
        return null;
    }
}
