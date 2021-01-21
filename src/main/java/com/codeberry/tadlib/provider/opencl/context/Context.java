package com.codeberry.tadlib.provider.opencl.context;

import com.codeberry.tadlib.memorymanagement.AbstractDisposer;
import com.codeberry.tadlib.memorymanagement.DisposalRegister;
import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.opencl.OpenCL;
import com.codeberry.tadlib.provider.opencl.device.Device;
import com.codeberry.tadlib.provider.opencl.kernel.Kernel;
import com.codeberry.tadlib.provider.opencl.kernel.KernelCollector;
import com.codeberry.tadlib.provider.opencl.program.Program;
import com.codeberry.tadlib.provider.opencl.queue.CommandQueue;
import com.codeberry.tadlib.util.ClockTimer;
import com.sun.jna.Pointer;

import java.util.*;

import static com.codeberry.tadlib.provider.opencl.consts.ErrorCode.throwOnError;
import static com.codeberry.tadlib.provider.opencl.device.Device.mapDevicePointers;
import static com.codeberry.tadlib.provider.opencl.device.Device.useDevicePointers;
import static com.codeberry.tadlib.util.ClockTimer.timer;
import static java.util.Collections.unmodifiableMap;
import static java.util.stream.Collectors.toList;

public class Context extends Pointer {
    // Doc from https://www.khronos.org/registry/OpenCL//sdk/1.2/docs/man/xhtml/clSetKernelArg.html:
    //    [...] applications are strongly encouraged to make additional cl_kernel objects for kernel functions for each host thread [...]
    private final ThreadLocal<Map<String, Kernel>> THREAD_KERNELS = new ThreadLocal<>();

    private final List<Device> devices;
    private final List<CommandQueue> queues;
    private final List<Program> programs;
    private final Disposer disposer;

    private Context(Pointer context, List<Device> devices, List<CommandQueue> queues, List<Program> programs) {
        super(Pointer.nativeValue(context));
        this.devices = devices;
        this.queues = queues;
        this.programs = programs;
        this.disposer = new Disposer(Pointer.nativeValue(context));
    }

    public Kernel findKernel(String kernelName) {
        Map<String, Kernel> kernels = ensureThreadLocalKernels();

        return kernels.get(kernelName);
    }

    private Map<String, Kernel> ensureThreadLocalKernels() {
        if (THREAD_KERNELS.get() == null) {
            System.out.println("First time accessing kernels from thread '" +
                    Thread.currentThread().getName() + "', creating kernels...");
            ClockTimer timer = timer("create kernels");

            Map<String, Kernel> kernels = createKernels(programs, devices);
            THREAD_KERNELS.set(kernels);

            System.out.println("...created " + kernels.size() + " kernels. " + timer);
        }

        return THREAD_KERNELS.get();
    }

    private static Map<String, Kernel> createKernels(List<Program> programs, List<Device> devices) {
        Map<String, Kernel> kernels = new HashMap<>();
        for (Program program : programs) {
            for (String k : program.getKernelNames()) {
                Kernel kernel = Kernel.create(program, k, devices);
                kernels.put(k, kernel);
            }
        }
        return unmodifiableMap(kernels);
    }

    public static Context create(List<Device> devices) {
        Pointer context = createContext(devices);
        List<CommandQueue> queues = devices.stream()
                .map(device -> CommandQueue.create(context, device))
                .collect(toList());

        List<Program> programs = new ArrayList<>();
        List<KernelCollector.ProgramSource> sources = KernelCollector.collectAllKernelSources();

        for (KernelCollector.ProgramSource ps : sources) {
            Program program = Program.create(context, devices, ps);
            programs.add(program);
        }

        Context r = new Context(context, devices, queues, programs);
        DisposalRegister.registerDisposer(r, r.disposer);
        return r;
    }

    public CommandQueue getQueue() {
        String name = ProviderStore.getDeviceName();
        if (name == null) {
            return queues.get(0);
        }

        for (CommandQueue queue : queues) {
            Device device = queue.getDevice();
            if (device.info.name.contains(name)) {
                return queue;
            }
        }

        throw new IllegalArgumentException("Unknown device: " + name);
    }

    private static Pointer createContext(List<Device> devices) {
        return mapDevicePointers(pointerArray ->
                throwOnError(errRef -> OpenCL.INSTANCE.clCreateContext(null,
                        devices.size(), pointerArray,
                        null, null, errRef)), devices);
    }

    @Override
    public String toString() {
        return "Context{" +
                "pointer=" + super.toString() +
                ", devices=" + devices.toString() +
                ", queues=" + queues.toString() +
                '}';
    }

    private static class Disposer extends AbstractDisposer {
        private final long nativeValue;

        public Disposer(long nativeValue) {
            this.nativeValue = nativeValue;
        }

        @Override
        protected void releaseResource() {
            throwOnError(() -> OpenCL.INSTANCE.clReleaseContext(Pointer.createConstant(nativeValue)));
        }

        @Override
        protected long getResourceId() {
            return nativeValue;
        }
    }

}
