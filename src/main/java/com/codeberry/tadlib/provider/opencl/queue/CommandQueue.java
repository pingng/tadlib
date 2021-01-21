package com.codeberry.tadlib.provider.opencl.queue;

import com.codeberry.tadlib.memorymanagement.AbstractDisposer;
import com.codeberry.tadlib.memorymanagement.DisposalRegister;
import com.codeberry.tadlib.provider.opencl.InProgressResources;
import com.codeberry.tadlib.provider.opencl.OpenCL;
import com.codeberry.tadlib.provider.opencl.SizeTArrayByReference;
import com.codeberry.tadlib.provider.opencl.device.Device;
import com.codeberry.tadlib.provider.opencl.kernel.Kernel;
import com.sun.jna.Pointer;

import static com.codeberry.tadlib.provider.opencl.SizeTArrayByReference.toSizeTArray;
import static com.codeberry.tadlib.provider.opencl.consts.ErrorCode.throwOnError;
import static java.lang.Math.min;

public class CommandQueue extends Pointer {
    private static final boolean FINISHED_AFTER_EACH_ENQUEUE = false;

    private static final boolean IS_ASYNC = true;
    /* cl_command_queue_properties - bitfield */
    private static final long CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE = (1 << 0);
    private static final long CL_QUEUE_PROFILING_ENABLE = (1 << 1);


    private final Device device;
    private final Disposer disposer;

    private CommandQueue(Pointer queue, Device device) {
        super(Pointer.nativeValue(queue));
        this.device = device;
        this.disposer = new Disposer(Pointer.nativeValue(queue));
    }

    public Device getDevice() {
        return device;
    }

    public static CommandQueue create(Pointer context, Device device) {
        Pointer queue = throwOnError(errCode -> OpenCL.INSTANCE.clCreateCommandQueue(context,
                device, IS_ASYNC ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : 0, errCode));

        CommandQueue r = new CommandQueue(queue, device);

        // Simple test that it works
        r.waitForFinish();

        DisposalRegister.registerDisposer(r, r.disposer);

        return r;
    }

    public void enqueueKernel(Kernel kernel, long actualGlobalItems,
                              WorkItemMode workItemMode,
                              InProgressResources resources) {
        long multiple = kernel.getInfo(getDevice()).preferredWorkGroupSizeMultiple;

        resources.useDependencyEvents(events -> throwOnError(() -> {
            long global = workItemMode.calcGlobalItems(actualGlobalItems, multiple);
            long local = multiple;

            SizeTArrayByReference global_work_size = resources.registerDisposableByRef(toSizeTArray(global));
            SizeTArrayByReference local_work_size = resources.registerDisposableByRef(toSizeTArray(local));

            return OpenCL.INSTANCE.wrappedEnqueueNDRangeKernel(this, kernel, 1, null,
                    global_work_size,
                    local_work_size,
                    events,
                    resources.opEvent);
        }));

        if (FINISHED_AFTER_EACH_ENQUEUE) {
            waitForFinish();
        }
    }

    public void enqueueKernel(Kernel kernel,
                              long[] actualGlobalItems,
                              long[] localItems,
                              InProgressResources resources) {
        SizeTArrayByReference global_work_size = resources.registerDisposableByRef(toSizeTArray(actualGlobalItems));
        SizeTArrayByReference local_work_size = resources.registerDisposableByRef(toSizeTArray(localItems));

        resources.useDependencyEvents(events -> {
            throwOnError(() -> OpenCL.INSTANCE.wrappedEnqueueNDRangeKernel(this, kernel, actualGlobalItems.length, null,
                    global_work_size,
                    local_work_size,
                    events,
                    resources.opEvent));
        });

        if (FINISHED_AFTER_EACH_ENQUEUE) {
            waitForFinish();
        }
    }

    public void waitForFinish() {
        throwOnError(() -> OpenCL.INSTANCE.clFinish(this));
    }

    public enum WorkItemMode {
        MULTIPLES_OF_PREFERRED_GROUP_SIZE;

        public long calcGlobalItems(long actualGlobalItems, long multiple) {
            if (actualGlobalItems <= multiple) {
                return multiple;
            } else {
                long groups = (actualGlobalItems + multiple - 1) / multiple;
                return multiple * groups;
            }
        }
    }

    private static class Disposer extends AbstractDisposer {
        private final long nativeValue;

        public Disposer(long nativeValue) {
            this.nativeValue = nativeValue;
        }

        @Override
        protected void releaseResource() {
            throwOnError(() -> OpenCL.INSTANCE.clReleaseCommandQueue(Pointer.createConstant(nativeValue)));
        }

        @Override
        protected long getResourceId() {
            return nativeValue;
        }
    }

}
