package com.codeberry.tadlib.provider.opencl.program;

import com.codeberry.tadlib.memorymanagement.AbstractDisposer;
import com.codeberry.tadlib.memorymanagement.DisposalRegister;
import com.codeberry.tadlib.provider.opencl.OpenCL;
import com.codeberry.tadlib.provider.opencl.SizeT;
import com.codeberry.tadlib.provider.opencl.SizeTByReference;
import com.codeberry.tadlib.provider.opencl.consts.ErrorCode;
import com.codeberry.tadlib.provider.opencl.device.Device;
import com.codeberry.tadlib.provider.opencl.kernel.KernelCollector;
import com.codeberry.tadlib.provider.opencl.ops.OclKernelSource;
import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import com.sun.jna.StringArray;

import java.util.List;

import static com.codeberry.tadlib.provider.opencl.consts.ErrorCode.throwOnError;
import static com.codeberry.tadlib.provider.opencl.device.Device.toPointerArray;

public class Program extends Pointer {
    /* cl_program_build_info */
    private static final int CL_PROGRAM_BUILD_STATUS = 0x1181;
    private static final int CL_PROGRAM_BUILD_OPTIONS = 0x1182;
    private static final int CL_PROGRAM_BUILD_LOG = 0x1183;
    private static final int CL_PROGRAM_BINARY_TYPE = 0x1184;

    public final Class<? extends OclKernelSource> programClass;
    private final List<String> kernelNames;
    private final Disposer disposer;

    private Program(Class<? extends OclKernelSource> programClass, List<String> kernelNames, Pointer program) {
        super(Pointer.nativeValue(program));
        this.programClass = programClass;
        this.kernelNames = kernelNames;
        this.disposer = new Disposer(Pointer.nativeValue(program));
    }


    public static Program create(Pointer context, List<Device> devices, KernelCollector.ProgramSource programSource) {
        try {
            OpenCL inst = OpenCL.INSTANCE;
            Pointer program = throwOnError(errCode -> inst.clCreateProgramWithSource(context, 1,
                    new StringArray(new String[]{programSource.source}), null, errCode));
            throwOnError(() -> inst.clBuildProgram(program,
                    devices.size(), toPointerArray(devices), null, null, null),
                    () -> getErrorMessage(program, devices));

            Program ret = new Program(programSource.programClass, programSource.kernels, program);

            DisposalRegister.registerDisposer(ret, ret.disposer);
            return ret;
        } catch (ErrorCode.OpenCLException e) {
            throw new OpenCLBuildException("While building & creating kernels for: " + programSource.programClass, e);
        }
    }

    private static String getErrorMessage(Pointer program, List<Device> devices) {
        StringBuilder buf = new StringBuilder();

        for (Device device : devices) {
            SizeTByReference sizeRet = new SizeTByReference();

            throwOnError(() -> OpenCL.INSTANCE.clGetProgramBuildInfo(program, device,CL_PROGRAM_BUILD_LOG,
                    null, null, sizeRet));

            Memory charBuf = new Memory(sizeRet.getValue());
            throwOnError(() -> OpenCL.INSTANCE.clGetProgramBuildInfo(program, device,CL_PROGRAM_BUILD_LOG,
                    new SizeT(sizeRet.getValue()), charBuf, null));

            buf.append("\n")
                    .append(device.info.name).append(":\n")
                    .append(charBuf.getString(0, "UTF-8"));
        }

        return buf.toString();
    }

    public List<String> getKernelNames() {
        return kernelNames;
    }

    private static class Disposer extends AbstractDisposer {
        private final long nativeValue;

        public Disposer(long nativeValue) {
            this.nativeValue = nativeValue;
        }

        @Override
        protected void releaseResource() {
            throwOnError(() -> OpenCL.INSTANCE.clReleaseProgram(Pointer.createConstant(nativeValue)));
        }

        @Override
        protected long getResourceId() {
            return nativeValue;
        }
    }

    private static class OpenCLBuildException extends ErrorCode.OpenCLException {
        OpenCLBuildException(String msg, Exception cause) {
            super(msg, cause);
        }
    }
}
