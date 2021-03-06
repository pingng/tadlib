package com.codeberry.tadlib.provider.opencl.kernel;

import com.codeberry.tadlib.memorymanagement.AbstractDisposer;
import com.codeberry.tadlib.memorymanagement.DisposalRegister;
import com.codeberry.tadlib.provider.opencl.*;
import com.codeberry.tadlib.provider.opencl.consts.ErrorCode;
import com.codeberry.tadlib.provider.opencl.device.Device;
import com.codeberry.tadlib.provider.opencl.device.DeviceType;
import com.codeberry.tadlib.provider.opencl.program.Program;
import com.codeberry.tadlib.provider.opencl.queue.CommandQueue;
import com.sun.jna.Pointer;

import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

import static com.codeberry.tadlib.provider.opencl.OclDataType.*;
import static com.codeberry.tadlib.provider.opencl.OpenCLHelper.InfoMarshaller.*;
import static com.codeberry.tadlib.provider.opencl.OpenCLHelper.getKernelWorkGroupInfoPointer;
import static com.codeberry.tadlib.provider.opencl.SizeT.*;
import static com.codeberry.tadlib.provider.opencl.consts.ErrorCode.throwOnError;
import static com.codeberry.tadlib.provider.opencl.kernel.KernelWorkGroupInfoCode.*;
import static java.util.stream.Collectors.*;

public class Kernel extends Pointer {
    private final Map<Device, Info> infoMap;
    private final Disposer disposer;

    private Kernel(Pointer kernel, Map<Device, Info> infoMap) {
        super(Pointer.nativeValue(kernel));
        this.infoMap = infoMap;
        this.disposer = new Disposer(Pointer.nativeValue(kernel));
    }

    public Info getInfo(Device device) {
        return infoMap.get(device);
    }

    public static Kernel create(Program program, String kernelName, List<Device> devices) {
        try {
            Pointer kernel = throwOnError(errCode -> OpenCL.INSTANCE.clCreateKernel(program, kernelName, errCode));

            Map<Device, Info> infoMap = new IdentityHashMap<>();
            for (Device device : devices) {
                Info info = getInfo(kernel, device);
                infoMap.put(device, info);
            }

            Kernel ret = new Kernel(kernel, infoMap);
            DisposalRegister.registerDisposer(ret, ret.disposer);
            return ret;
        } catch (ErrorCode.OpenCLException e) {
            throw new OpenCLKernelBuildException("While creating kernel: " + kernelName + " from class '" + program.programClass.getSimpleName() + "'", e);
        }
    }

    private static Info getInfo(Pointer kernel, Device device) {
        long[] kernelWorkGroupInfoPointer;
        if (device.info.type == DeviceType.CL_DEVICE_TYPE_CUSTOM) {
            kernelWorkGroupInfoPointer = getKernelWorkGroupInfoPointer(kernel, device, CL_KERNEL_GLOBAL_WORK_SIZE, RETURN_SIZE_T_ARRAY(3));
        } else {
            kernelWorkGroupInfoPointer = new long[]{-1, -1, -1};
        }

        return new Info(
                getKernelWorkGroupInfoPointer(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, RETURN_LONG),
                getKernelWorkGroupInfoPointer(kernel, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, RETURN_SIZE_T_ARRAY(3)),
                getKernelWorkGroupInfoPointer(kernel, device, CL_KERNEL_LOCAL_MEM_SIZE, RETURN_LONG),
                getKernelWorkGroupInfoPointer(kernel, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, RETURN_SIZE_T),
                getKernelWorkGroupInfoPointer(kernel, device, CL_KERNEL_PRIVATE_MEM_SIZE, RETURN_LONG),
                kernelWorkGroupInfoPointer
        );
    }

    public ArgSetter createArgSetter(InProgressResources resources) {
        return new ArgSetter(this, resources);
    }

    public long getPreferredWorkGroupSizeMultiple(CommandQueue queue) {
        return getInfo(queue.getDevice()).preferredWorkGroupSizeMultiple;
    }

    public int alignWithPreferredWorkGroupSizeMultiple(CommandQueue queue, int value) {
        long multiple = getPreferredWorkGroupSizeMultiple(queue);
        return (int) ((value + multiple - 1) / multiple * multiple);
    }

    public static class ArgSetter {
        private final Kernel owner;
        private final InProgressResources resources;
        private int paramIndex;

        public ArgSetter(Kernel owner, InProgressResources resources) {
            this.owner = owner;
            this.resources = resources;
        }

        public ArgSetter nextArg(OclArray oclArray) {
            resources.registerDependency(oclArray);
            Pointer pointer = (oclArray != null ? oclArray.getArgPointer() : null);
            return nextArg(pointer);
        }

        public ArgSetter nextArg(OclIntArray oclIntArray) {
            resources.registerDependency(oclIntArray);
            Pointer pointer = (oclIntArray != null ? oclIntArray.getArgPointer() : null);
            return nextArg(pointer);
        }

        public ArgSetter nextArgDisposable(OclBuffer buffer) {
            resources.registerDisposableBuffer(buffer);
            return nextArg(buffer.argPointer);
        }

        public ArgSetter nextArgKeepRef(OclBuffer buffer) {
            resources.registerReferredBuffer(buffer);
            return nextArg((buffer != null ? buffer.argPointer : null));
        }

        private ArgSetter nextArg(Pointer arg) {
            throwOnError(() -> OpenCL.INSTANCE.clSetKernelArg(owner, paramIndex++,
                    POINTER_SIZE_T, arg));

            return this;
        }

        public ArgSetter nextArg(int[] intArr) {
            throwOnError(() -> OpenCL.INSTANCE.clSetKernelArg(owner, paramIndex++,
                    POINTER_SIZE_T, resources.registerReadOnlyArg(intArr)));

            return this;
        }

        public ArgSetter nextArg(long[] longArr) {
            throwOnError(() -> OpenCL.INSTANCE.clSetKernelArg(owner, paramIndex++,
                    POINTER_SIZE_T, resources.registerReadOnlyArg(longArr)));

            return this;
        }

        public ArgSetter nextArg(double[] doubleArr) {
            throwOnError(() -> OpenCL.INSTANCE.clSetKernelArg(owner, paramIndex++,
                    POINTER_SIZE_T, resources.registerReadOnlyArg(doubleArr)));

            return this;
        }

        public ArgSetter nextArg(int v) {
            throwOnError(() -> OpenCL.INSTANCE.clSetKernelArg(owner, paramIndex++,
                    new SizeT(cl_int.sizeOfElements(1)),
                    resources.argInt(v)));

            return this;
        }

        public ArgSetter nextArg(double v) {
            throwOnError(() -> OpenCL.INSTANCE.clSetKernelArg(owner, paramIndex++,
                    new SizeT(cl_double.sizeOfElements(1)),
                    resources.argDouble(v)));

            return this;
        }

        public ArgSetter nextArg(long v) {
            throwOnError(() -> OpenCL.INSTANCE.clSetKernelArg(owner, paramIndex++,
                    new SizeT(cl_long.sizeOfElements(1)),
                    resources.argLong(v)));

            return this;
        }

        public ArgSetter nextArg(boolean v) {
            throwOnError(() -> OpenCL.INSTANCE.clSetKernelArg(owner, paramIndex++,
                    new SizeT(cl_int.sizeOfElements(1)),
                    resources.argBoolean(v)));

            return this;
        }

        public ArgSetter nextArgLocalDoubles(int size) {
            throwOnError(() -> OpenCL.INSTANCE.clSetKernelArg(owner, paramIndex++,
                    new SizeT(cl_double.sizeOfElements(size)),
                    null));

            return this;
        }

        public ArgSetter nextArgLocalInts(int size) {
            throwOnError(() -> OpenCL.INSTANCE.clSetKernelArg(owner, paramIndex++,
                    new SizeT(cl_int.sizeOfElements(size)),
                    null));

            return this;
        }
    }

    public static class Info {
        // size_t
        public final long workGroupSize;
        // size_t[3]
        public final List<Long> compileWorkGroupSize;
        // cl_ulong
        public final long localMemSize;
        // size_t
        public final long preferredWorkGroupSizeMultiple;
        // cl_ulong
        public final long privateMemSize;
        // size_t[3]
        public final List<Long> globalWorkSize;

        public Info(long workGroupSize, long[] compileWorkGroupSize, long localMemSize, long preferredWorkGroupSizeMultiple, long privateMemSize, long[] globalWorkSize) {
            this.workGroupSize = workGroupSize;
            this.compileWorkGroupSize = toList(compileWorkGroupSize);
            this.localMemSize = localMemSize;
            this.preferredWorkGroupSizeMultiple = preferredWorkGroupSizeMultiple;
            this.privateMemSize = privateMemSize;
            this.globalWorkSize = toList(globalWorkSize);
        }

        private static List<Long> toList(long[] compileWorkGroupSize) {
            return Arrays.stream(compileWorkGroupSize).boxed().collect(toUnmodifiableList());
        }

        @Override
        public String toString() {
            return "Info{" +
                    "workGroupSize=" + workGroupSize +
                    ", compileWorkGroupSize=" + compileWorkGroupSize +
                    ", localMemSize=" + localMemSize +
                    ", preferredWorkGroupSizeMultiple=" + preferredWorkGroupSizeMultiple +
                    ", privateMemSize=" + privateMemSize +
                    ", globalWorkSize=" + globalWorkSize +
                    '}';
        }
    }

    private static class Disposer extends AbstractDisposer {
        private final long nativeValue;

        public Disposer(long nativeValue) {
            this.nativeValue = nativeValue;
        }

        @Override
        protected void releaseResource() {
            throwOnError(() -> OpenCL.INSTANCE.clReleaseKernel(Pointer.createConstant(nativeValue)));
        }

        @Override
        protected long getResourceId() {
            return nativeValue;
        }
    }

    private static class OpenCLKernelBuildException extends ErrorCode.OpenCLException {
        OpenCLKernelBuildException(String msg, Exception cause) {
            super(msg, cause);
        }
    }

}
