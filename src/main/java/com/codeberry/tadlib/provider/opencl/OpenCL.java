package com.codeberry.tadlib.provider.opencl;

import com.codeberry.tadlib.provider.opencl.jna.TADMemory;
import com.codeberry.tadlib.provider.opencl.jna.TADPointerByReference;
import com.codeberry.tadlib.memorymanagement.LeakDetector;
import com.codeberry.tadlib.provider.opencl.device.Device;
import com.codeberry.tadlib.provider.opencl.device.DevicesPointer;
import com.codeberry.tadlib.provider.opencl.kernel.Kernel;
import com.codeberry.tadlib.provider.opencl.platform.Platform;
import com.codeberry.tadlib.provider.opencl.platform.PlatformsPointer;
import com.sun.jna.*;
import com.sun.jna.ptr.IntByReference;

public class OpenCL {

    public static final OpenCL INSTANCE = new OpenCL();

    static {
        Native.register("opencl");
    }

    public native int clGetPlatformIDs(int num_entries, PlatformsPointer platforms, IntByReference num_platforms);

    public native int clGetPlatformInfo(Pointer platform, int param_name, int param_value_size, Pointer param_value, Pointer param_value_size_ret);

    public native int clGetDeviceIDs(Platform platform, NativeLong device_type, int num_entries, DevicesPointer devices, IntByReference num_devices);

    public native int clGetDeviceInfo(Pointer device, int param_name, SizeT param_value_size, Pointer param_value, SizeTByReference param_value_size_ret);

    public native int clGetKernelWorkGroupInfo(Pointer kernel, Device device, int param_name, SizeT param_value_size, Pointer param_value, SizeTByReference param_value_size_ret);

    public native Pointer clCreateContext(Pointer properties, int num_devices, PointerArray devices, Pointer pfn_notify, Pointer user_data, IntByReference errcode_ret);

    public native int clReleaseContext(Pointer context);

    public native int clGetContextInfo(Pointer context, int param_name, SizeT param_value_size, Pointer param_value, Pointer param_value_size_ret);

    public native Pointer clCreateCommandQueue(Pointer context, Pointer device, long properties, IntByReference errcode_ret);

    public native int clReleaseCommandQueue(Pointer command_queue);

    /**
     * @deprecated use wrapper
     */
    public native Pointer clCreateBuffer(Pointer context, long flags, SizeT size, Pointer host_ptr, IntByReference errcode_ret);
    public Pointer wrapperCreateBuffer(Pointer context, long flags, SizeT size, Pointer host_ptr, IntByReference errcode_ret) {
        Pointer pointer = clCreateBuffer(context, flags, size, host_ptr, errcode_ret);
        LeakDetector.allocate(Pointer.nativeValue(pointer));
        return pointer;
    }

    /**
     * @deprecated use wrapper
     */
    public native int clRetainMemObject(Pointer memobj);
    public int wrapperRetainMemObject(Pointer memobj) {
        LeakDetector.allocate(Pointer.nativeValue(memobj));

        return clRetainMemObject(memobj);
    }

    /**
     * @deprecated use wrapper
     */
    public native int clReleaseMemObject(Pointer memobj);
    public int wrapperReleaseMemObject(Pointer memobj) {
        LeakDetector.release(Pointer.nativeValue(memobj));

        return clReleaseMemObject(memobj);
    }

    /**
     * @deprecated use the wrapper
     */
    public native int clEnqueueWriteBuffer(Pointer command_queue, Pointer oclPointer, boolean blocking_write, SizeT offset, SizeT size, Pointer ptr, int num_events_in_wait_list,
                                           PointerArray event_wait_list, TADPointerByReference event);

    public int wrapperEnqueueWriteBuffer(Pointer command_queue, Pointer oclPointer, boolean blocking_write,
                                         SizeT offset, SizeT size, Pointer ptr,
                                         PointerArray event_wait_list, OclEventByReference event) {
        int r = clEnqueueWriteBuffer(command_queue, oclPointer, blocking_write,
                offset, size, ptr,
                (event_wait_list == null ? 0 : event_wait_list.length()), event_wait_list,
                event);
        if (event != null) {
            event.onFinishedCall();
        }
        return r;
    }

    /**
     * @deprecated use wrapper
     */
    public native int clEnqueueReadBuffer(Pointer command_queue, Pointer oclPointer, boolean blocking_read, SizeT offset, SizeT size, Pointer ptr, int num_events_in_wait_list,
                                          PointerArray event_wait_list, TADPointerByReference event);

    public int wrapperEnqueueReadBuffer(Pointer command_queue, Pointer oclPointer,
                                        boolean blocking_read,
                                        SizeT offset, SizeT size, Pointer ptr,
                                        PointerArray event_wait_list,
                                        OclEventByReference event) {
        int r = clEnqueueReadBuffer(command_queue, oclPointer,
                blocking_read, offset, size, ptr,
                (event_wait_list == null ? 0 : event_wait_list.length()),
                event_wait_list, event);
        if (event != null) {
            event.onFinishedCall();
        }

        return r;
    }

    public native Pointer clCreateProgramWithSource(Pointer context, int count, StringArray strings, Pointer lengths, IntByReference errcode_ret);

    public native int clReleaseProgram(Pointer program);

    public native int clBuildProgram(Pointer program, int num_devices, PointerArray device_list, String options, Callback pfn_notify, Pointer user_data);

    public native int clGetProgramBuildInfo(Pointer program,
                                            Device device,
                                            int param_name,
                                            SizeT param_value_size,
                                            Pointer param_value, SizeTByReference param_value_size_ret);

    public native Pointer clCreateKernel(Pointer program, String kernel_name, IntByReference errcode_ret);

    public native int clReleaseKernel(Pointer kernel);

    public native int clSetKernelArg(Kernel kernel, int arg_index, SizeT arg_size, Pointer arg_value);

    /**
     * @param event_wait_list The memory associated with event_wait_list can be reused or freed after the function returns.
     * @deprecated use the wrapper
     */
    public native int clEnqueueNDRangeKernel(Pointer command_queue, Pointer kernel,
                                             int work_dim,
                                             SizeTArrayByReference global_work_offset,
                                             SizeTArrayByReference global_work_size,
                                             SizeTArrayByReference local_work_size,
                                             int num_events_in_wait_list,
                                             PointerArray event_wait_list,
                                             TADPointerByReference event);

    public int wrappedEnqueueNDRangeKernel(Pointer command_queue, Pointer kernel,
                                           int work_dim,
                                           SizeTArrayByReference global_work_offset,
                                           SizeTArrayByReference global_work_size,
                                           SizeTArrayByReference local_work_size,
                                           PointerArray event_wait_list,
                                           OclEventByReference event) {
        int ret = clEnqueueNDRangeKernel(command_queue, kernel, work_dim,
                global_work_offset, global_work_size, local_work_size,
                (event_wait_list == null ? 0 : event_wait_list.length()),
                event_wait_list, event);
        if (event != null) {
            event.onFinishedCall();
        }
        return ret;
    }

    public native int clReleaseEvent(Pointer event);

    public native int clWaitForEvents(int num_events, PointerArray event_list);

    /**
     * @deprecated use wrapper
     */
    public native int clEnqueueFillBuffer(Pointer command_queue,
                                          Pointer buffer,
                                          Pointer pattern,
                                          SizeT pattern_size,
                                          SizeT offset,
                                          SizeT size,
                                          int num_events_in_wait_list,
                                          PointerArray event_wait_list,
                                          OclEventByReference event);

    public int wrapperEnqueueFillBuffer(Pointer command_queue,
                                        Pointer buffer,
                                        Pointer pattern,
                                        SizeT pattern_size,
                                        SizeT offset,
                                        SizeT size,
                                        PointerArray event_wait_list,
                                        OclEventByReference event) {
        int r = clEnqueueFillBuffer(command_queue, buffer,
                pattern, pattern_size,
                offset, size,
                (event_wait_list == null ? 0 : event_wait_list.length()),
                event_wait_list, event);
        if (event != null) {
            event.onFinishedCall();
        }
        return r;
    }

    /**
     * @deprecated use wrapper
     */
    public native int clEnqueueCopyBuffer(Pointer command_queue,
                                          Pointer src_buffer,
                                          Pointer dst_buffer,
                                          SizeT src_offset,
                                          SizeT dst_offset,
                                          SizeT size,
                                          int num_events_in_wait_list,
                                          PointerArray event_wait_list,
                                          TADPointerByReference event);

    public int wrapperEnqueueCopyBuffer(Pointer command_queue,
                                        Pointer src_buffer,
                                        Pointer dst_buffer,
                                        SizeT src_offset,
                                        SizeT dst_offset,
                                        SizeT size,
                                        PointerArray event_wait_list,
                                        OclEventByReference event) {
        int r = clEnqueueCopyBuffer(command_queue, src_buffer, dst_buffer, src_offset, dst_offset, size,
                (event_wait_list == null ? 0 : event_wait_list.length()),
                event_wait_list, event);
        if (event != null) {
            event.onFinishedCall();
        }
        return r;
    }


    public native int clFinish(Pointer command_queue);

    // FROM: com.sun.jna.Function
    public static class PointerArray extends TADMemory {
        private final int length;

        private PointerArray(Pointer[] arg) {
            super((long) Native.POINTER_SIZE * (arg.length + 1));
            this.length = arg.length;
            for (int i = 0; i < arg.length; i++) {
                setPointer((long) i * Native.POINTER_SIZE, arg[i]);
            }
            setPointer((long) Native.POINTER_SIZE * arg.length, null);
        }

        public static <R> R mapPointerArray(java.util.function.Function<PointerArray, R> mapper, Pointer... pointers) {
            PointerArray pa = new PointerArray(pointers);
            try {
                return mapper.apply(pa);
            } finally {
                pa.dispose();
            }
        }

        public static void usePointerArray(java.util.function.Consumer<PointerArray> consumer, Pointer... pointers) {
            Pointer[] actual = extractNonNulls(pointers);
            PointerArray pa = (actual != null ? new PointerArray(actual) : null);
            try {
                consumer.accept(pa);
            } finally {
                if (pa != null) {
                    pa.dispose();
                }
            }
        }

        private static Pointer[] extractNonNulls(Pointer[] pointers) {
            if (pointers == null) {
                return null;
            }
            int nonNulls = 0;
            for (Pointer p : pointers) {
                if (p != null) {
                    nonNulls++;
                }
            }
            if (nonNulls == 0) {
                return null;
            } else if (nonNulls == pointers.length) {
                return pointers;
            }

            Pointer[] filtered = new Pointer[nonNulls];
            for (int i = 0, f = 0; i < pointers.length; i++) {
                Pointer p = pointers[i];
                if (p != null) {
                    filtered[f++] = p;
                }
            }
            return filtered;
        }

        public int length() {
            return length;
        }
    }
}