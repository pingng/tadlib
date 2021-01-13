package com.codeberry.tadlib.provider.opencl;

import com.codeberry.tadlib.provider.opencl.buffer.BufferMemFlags;
import com.codeberry.tadlib.provider.opencl.context.Context;
import com.codeberry.tadlib.provider.opencl.device.Device;
import com.codeberry.tadlib.provider.opencl.device.DeviceInfoCode;
import com.codeberry.tadlib.provider.opencl.device.DeviceType;
import com.codeberry.tadlib.provider.opencl.device.DevicesPointer;
import com.codeberry.tadlib.provider.opencl.kernel.Kernel;
import com.codeberry.tadlib.provider.opencl.kernel.KernelWorkGroupInfoCode;
import com.codeberry.tadlib.provider.opencl.platform.Platform;
import com.codeberry.tadlib.provider.opencl.platform.PlatformInfoCode;
import com.codeberry.tadlib.provider.opencl.platform.PlatformsPointer;
import com.sun.jna.Memory;
import com.sun.jna.NativeLong;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.IntByReference;

import java.util.ArrayList;
import java.util.List;

import static com.codeberry.tadlib.provider.opencl.consts.ErrorCode.*;

public class OpenCLHelper {
    static List<Platform> getPlatforms() {
        OpenCL inst = OpenCL.INSTANCE;

        IntByReference _platformCount = new IntByReference();
        throwOnError(() -> inst.clGetPlatformIDs(0, null, _platformCount));

        int platformCount = _platformCount.getValue();
        PlatformsPointer platformsPointer = new PlatformsPointer(platformCount);
        throwOnError(() -> inst.clGetPlatformIDs(platformCount, platformsPointer, null));

        List<Platform> r = new ArrayList<>();
        for (int i = 0; i < platformCount; i++) {
            r.add(Platform.fromPlatformsIndex(platformsPointer, i));
        }
        return r;
    }

    public static String getPlatformInfoString(Pointer platform, PlatformInfoCode infoCode) {
        Memory buf = new Memory(512);
        Pointer sizeBuf = new Memory(4);

        throwOnError(() -> OpenCL.INSTANCE.clGetPlatformInfo(platform,
                infoCode.code,
                512,
                buf,
                sizeBuf));

        return buf.getString(0);
    }

    static List<Device> getDevices(Platform platform, DeviceType... types) {
        NativeLong bits = DeviceType.toBits(types);

        OpenCL inst = OpenCL.INSTANCE;

        IntByReference _count = new IntByReference();
        throwOnError(() -> inst.clGetDeviceIDs(platform, bits, 0, null, _count));

        int count = _count.getValue();
        DevicesPointer devices = new DevicesPointer(count);
        throwOnError(() -> inst.clGetDeviceIDs(platform, bits, count, devices, null));

        List<Device> r = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            r.add(Device.fromDevicesIndex(devices, i));
        }
        return r;
    }

    public static String getDeviceInfoString(Pointer pointer, DeviceInfoCode code) {
        Pointer _buf = getDeviceInfoPointer(pointer, code);

        return _buf.getString(0);
    }

    public static long getDeviceInfoLong(Pointer pointer, DeviceInfoCode code) {
        Pointer _buf = getDeviceInfoPointer(pointer, code);

        return _buf.getLong(0);
    }

    public static int getDeviceInfoInt(Pointer pointer, DeviceInfoCode code) {
        Pointer _buf = getDeviceInfoPointer(pointer, code);

        return _buf.getInt(0);
    }

    public static SizeT getDeviceInfoSizeT(Pointer pointer, DeviceInfoCode code) {
        Pointer _buf = getDeviceInfoPointer(pointer, code);

        return SizeT.getSizeT(_buf, 0);
    }

    public static DeviceType getDeviceInfoType(Pointer pointer, DeviceInfoCode code) {
        Pointer _buf = getDeviceInfoPointer(pointer, code);

        long bits = _buf.getLong(0);

        return DeviceType.fromBits(bits);
    }

    public static Pointer getDeviceInfoPointer(Pointer pointer, DeviceInfoCode code) {
        return getInfoPointer((paramValueSize, paramValue, paramValueSizeRet) ->
                OpenCL.INSTANCE.clGetDeviceInfo(pointer, code.code, paramValueSize, paramValue, paramValueSizeRet));
    }

    public static <R> R getKernelWorkGroupInfoPointer(Pointer kernel, Device device, KernelWorkGroupInfoCode code, InfoMarshaller<R> marshaller) {
        Pointer pointer = getInfoPointer((paramValueSize, paramValue, paramValueSizeRet) ->
                OpenCL.INSTANCE.clGetKernelWorkGroupInfo(kernel, device, code.code, paramValueSize, paramValue, paramValueSizeRet));
        return marshaller.getValue(pointer);
    }

    public static abstract class InfoMarshaller<R> {
        public static final InfoMarshaller<Long> RETURN_LONG = new InfoMarshaller<>() {
            @Override
            Long getValue(Pointer pointer) {
                return pointer.getLong(0);
            }
        };
        public static InfoMarshaller<long[]> RETURN_SIZE_T_ARRAY(int length) {
            return new InfoMarshaller<>() {
                @Override
                long[] getValue(Pointer pointer) {
                    long[] r = new long[length];
                    for (int i = 0; i < r.length; i++) {
                        r[i] = SizeT.getSizeT(pointer, i).longValue();
                    }
                    return r;
                }
            };
        };
        public static final InfoMarshaller<Long> RETURN_SIZE_T = new InfoMarshaller<>() {
            @Override
            Long getValue(Pointer pointer) {
                return SizeT.getSizeT(pointer, 0).longValue();
            }
        };

        abstract R getValue(Pointer pointer);
    }

    private static Pointer getInfoPointer(GetInfoCall infoCall) {
        SizeT sizeT = new SizeT();

        SizeTByReference _size = new SizeTByReference();
        throwOnError(() -> infoCall.getInfo(sizeT, null, _size));
        long len = _size.getValue();

        Pointer _buf = new Memory(len);
        sizeT.setValue(len);
        throwOnError(() -> infoCall.getInfo(sizeT, _buf, _size));
        return _buf;
    }

    private interface GetInfoCall {
        int getInfo(SizeT paramValueSize, Pointer paramValue, SizeTByReference paramValueSizeRet);
    }

    static Context createContext(List<Device> devices) {
//        System.out.println("_errCode = " + _errCode.getInt(0));
//        System.out.println("Context ref count: " + getContextInfo(context, OpenCLContextInfo.CL_CONTEXT_REFERENCE_COUNT));
//        System.out.println("Context device count: " + getContextInfo(context, OpenCLContextInfo.CL_CONTEXT_NUM_DEVICES));
//        System.out.println("Context devices: " +
//                Arrays.toString(getContextInfo(context, OpenCLContextInfo.CL_CONTEXT_DEVICES)));

        // Create a command queue
//        Li
//        Pointer command_queue = throwOnError(errCode -> inst.clCreateCommandQueue(context, deviceId, 0, errCode));

        return Context.create(devices);
    }

}
