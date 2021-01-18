package com.codeberry.tadlib.provider.opencl;

import com.codeberry.tadlib.provider.opencl.buffer.BufferMemFlags;
import com.codeberry.tadlib.provider.opencl.context.Context;
import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.ByReference;
import com.sun.jna.ptr.DoubleByReference;
import com.sun.jna.ptr.IntByReference;
import com.sun.jna.ptr.LongByReference;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Stream;

import static com.codeberry.tadlib.provider.opencl.OclBuffer.ByteSize.sizeOf;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.createBuffer;
import static com.codeberry.tadlib.provider.opencl.OclDataType.*;
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
    private List<Memory> memoryPointers = new ArrayList<>();
    private List<ByReference> references = new ArrayList<>();
    private List<OclEventByReference> events = new ArrayList<>();
    private List<OclBuffer> disposableBuffers = new ArrayList<>();
    private List<OclBuffer> strongReferredBuffers = new ArrayList<>();
    private List<OclArray> strongReferredNDArrays = new ArrayList<>();
    private List<OclIntArray> strongReferredNDIntArrays = new ArrayList<>();

    public InProgressResources(Context context) {
        this.context = context;

        opEvent = new OclEventByReference();
        events.add(opEvent);
    }

    public void registerDependency(OclArray dependantArray) {
        this.strongReferredNDArrays.add(dependantArray);
    }

    public void registerDependency(OclIntArray dependantArray) {
        this.strongReferredNDIntArrays.add(dependantArray);
    }

    public void registerDisposableBuffer(OclBuffer dependantArray) {
//            (new Exception("Registered mem: " + Pointer.nativeValue(dependantArray))).printStackTrace();
        this.disposableBuffers.add(dependantArray);
    }

    public void registerReferredBuffer(OclBuffer keepRefToBuffer) {
//            (new Exception("Keep mem: " + Pointer.nativeValue(keepRefToBuffer))).printStackTrace();
        this.strongReferredBuffers.add(keepRefToBuffer);
    }

    public OpenCL.PointerArray getDependencyEvents() {
        Pointer[] events = toEventPointers(() -> Stream.concat(
                strongReferredNDArrays.stream().map(a -> a.resources),
                strongReferredNDIntArrays.stream().map(a -> a.resources)));
        if (events != null) {
            return OpenCL.PointerArray.pointers(events);
        }
        return null;
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
        LongByReference ref = new LongByReference(v);
        references.add(ref);

        return ref.getPointer();
    }

    public Pointer argInt(int v) {
        IntByReference ref = new IntByReference(v);
        references.add(ref);

        return ref.getPointer();
    }

    public Pointer argDouble(double v) {
        DoubleByReference ref = new DoubleByReference(v);
        references.add(ref);

        return ref.getPointer();
    }

    public Pointer argBoolean(boolean v) {
        IntByReference ref = new IntByReference(v ? 1 : 0);
        references.add(ref);

        return ref.getPointer();
    }

    public synchronized boolean isDisposed() {
        return opEvent == null;
    }

    public synchronized void disposeDeep() {
        for (OclArray ndArr : strongReferredNDArrays) {
            ndArr.resources.disposeDeep();
        }
        for (OclIntArray ndArr : strongReferredNDIntArrays) {
            ndArr.resources.disposeDeep();
        }
        for (OclEventByReference event : events) {
            event.oclRelease();
        }
        for (OclBuffer buf : disposableBuffers) {
            buf.dispose();
        }

        strongReferredNDArrays.clear();
        strongReferredNDIntArrays.clear();
        strongReferredBuffers.clear();
        disposableBuffers.clear();
        memoryPointers.clear();
        references.clear();
        events.clear();
        opEvent = null;
    }

    public Pointer registerReadOnlyArg(int[] v) {
        if (v != null) {
            Memory memory = createMemory(v);
            OclBuffer buf = createBuffer(context, sizeOf(cl_int, v.length), BufferMemFlags.CL_MEM_READ_ONLY);
            OclEventByReference bufferWriteEvt = new OclEventByReference();
            throwOnError(() -> OpenCL.INSTANCE.wrapperEnqueueWriteBuffer(
                    context.getQueue(), buf, false, new SizeT(0), new SizeT(memory.size()),
                    memory, null, bufferWriteEvt));

            Memory pointerToPointer = new Memory(Native.POINTER_SIZE);
            pointerToPointer.setPointer(0, buf);

            memoryPointers.add(memory);
            memoryPointers.add(pointerToPointer);
            events.add(bufferWriteEvt);

            registerDisposableBuffer(buf);

            return pointerToPointer;
        }
        return Pointer.NULL;
    }

    public Pointer registerReadOnlyArg(long[] v) {
        if (v != null) {
            Memory memory = createMemory(v);
            OclBuffer buf = createBuffer(context, sizeOf(cl_long, v.length), BufferMemFlags.CL_MEM_READ_ONLY);
            OclEventByReference bufferWriteEvt = new OclEventByReference();
            throwOnError(() -> OpenCL.INSTANCE.wrapperEnqueueWriteBuffer(
                    context.getQueue(), buf, false, new SizeT(0), new SizeT(memory.size()),
                    memory, null, bufferWriteEvt));

            Memory pointerToPointer = new Memory(Native.POINTER_SIZE);
            pointerToPointer.setPointer(0, buf);

            memoryPointers.add(memory);
            memoryPointers.add(pointerToPointer);
            events.add(bufferWriteEvt);

            registerDisposableBuffer(buf);

            return pointerToPointer;
        }
        return Pointer.NULL;
    }

    public Pointer registerReadOnlyArg(double[] v) {
        if (v != null) {
            Memory memory = createMemory(v);
            OclBuffer buf = createBuffer(context, sizeOf(cl_double, v.length), BufferMemFlags.CL_MEM_READ_ONLY);
            OclEventByReference bufferWriteEvt = new OclEventByReference();
            throwOnError(() -> OpenCL.INSTANCE.wrapperEnqueueWriteBuffer(
                    context.getQueue(), buf, false, new SizeT(0), new SizeT(memory.size()),
                    memory, null, bufferWriteEvt));

            Memory pointerToPointer = new Memory(Native.POINTER_SIZE);
            pointerToPointer.setPointer(0, buf);

            memoryPointers.add(memory);
            memoryPointers.add(pointerToPointer);
            events.add(bufferWriteEvt);

            registerDisposableBuffer(buf);

            return pointerToPointer;
        }
        return Pointer.NULL;
    }

    private static Memory createMemory(int[] intArr) {
        Memory dimBlockSizesLeftPointer = new Memory(cl_int.sizeOfElements(intArr.length));
        dimBlockSizesLeftPointer.write(0, intArr, 0, intArr.length);
        return dimBlockSizesLeftPointer;
    }

    private static Memory createMemory(long[] longArr) {
        Memory dimBlockSizesLeftPointer = new Memory(cl_long.sizeOfElements(longArr.length));
        dimBlockSizesLeftPointer.write(0, longArr, 0, longArr.length);
        return dimBlockSizesLeftPointer;
    }

    private static Memory createMemory(double[] doubleArr) {
        Memory dimBlockSizesLeftPointer = new Memory(cl_double.sizeOfElements(doubleArr.length));
        dimBlockSizesLeftPointer.write(0, doubleArr, 0, doubleArr.length);
        return dimBlockSizesLeftPointer;
    }

    public void registerDependencyEvent(OclEventByReference event) {
        Objects.requireNonNull(event, "Cannot add NULL dependency event");

        events.add(event);
    }

    String getContentStatus() {
        StringBuilder sb = new StringBuilder();
        tryReadBuffers("StrongRefBuf", sb, strongReferredBuffers);
        tryReadBuffers("DisposableRefBuf", sb, disposableBuffers);
        tryReadBuffers("NDArray.buffer", sb, extractBuffers(strongReferredNDArrays, arr -> arr.buffer));
        tryReadBuffers("NDIntArray.buffer", sb, extractBuffers(strongReferredNDIntArrays, arr -> arr.buffer));
        for (int i = 0; i < memoryPointers.size(); i++) {
            Memory m = memoryPointers.get(i);
            sb.append("Memory[").append(i).append("/").append(memoryPointers.size()).append("]:").append(m.size()).append(":readOk=");
            try {
                byte[] tmp = new byte[toIntExact(m.size())];
                m.read(0, tmp, 0, tmp.length);
                sb.append("true");
            } catch (Exception e) {
                sb.append("false").append(e.toString());
            }
            sb.append("\n");
        }
        for (int i = 0; i < references.size(); i++) {
            ByReference ref = references.get(i);
            sb.append("Reference[").append(i).append("/").append(references.size()).append("]:").append(ref).append("\n");
        }

        OpenCL.PointerArray events = getDependencyEvents();
        if (events != null) {
            for (int i = 0; i < events.length(); i++) {
                sb.append("Event[").append(i).append("/").append(events.length()).append("]:waitRet=")
                        .append(OpenCL.INSTANCE.clWaitForEvents(1, OpenCL.PointerArray.pointers(events.getPointer((long) i * Native.POINTER_SIZE))))
                        .append("\n");
            }
        }

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
