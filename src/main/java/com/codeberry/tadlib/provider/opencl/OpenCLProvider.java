package com.codeberry.tadlib.provider.opencl;

import com.codeberry.tadlib.provider.java.JavaShape;
import com.codeberry.tadlib.array.NDArray;
import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.array.util.MultiDimArrayFlattener;
import com.codeberry.tadlib.provider.Provider;
import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.opencl.context.Context;
import com.codeberry.tadlib.provider.opencl.device.Device;
import com.codeberry.tadlib.provider.opencl.device.DeviceType;
import com.codeberry.tadlib.provider.opencl.platform.Platform;

import java.util.ArrayList;
import java.util.List;

import static com.codeberry.tadlib.provider.opencl.OclArray.createNDArray;
import static com.codeberry.tadlib.provider.opencl.OpenCLHelper.getDevices;
import static com.codeberry.tadlib.provider.opencl.OpenCLHelper.getPlatforms;
import static java.util.stream.Collectors.toList;

public class OpenCLProvider implements Provider {

    public static final String PROP_TAD_OPENCL_DEVICES = "tad.opencl.devices";

    static {
        System.out.println(OpenCL.INSTANCE);
    }

    private final Context context;

    public OpenCLProvider() {
        this.context = createContext(0);
        System.out.println("context = " + context);
    }

    private Context createContext(int platformIndex) {
        OpenCL inst = OpenCL.INSTANCE;
        System.out.println("OpenCL.INSTANCE = " + inst);
        List<Platform> platforms = getPlatforms();
        System.out.println("platforms = " + platforms);
        Platform platform = platforms.get(platformIndex);

        List<Device> devices = getDevices(platform, DeviceType.CL_DEVICE_TYPE_ALL);

        System.out.println("Found devices:");
        for (int i = 0; i < devices.size(); i++) {
            System.out.println("  " + i + ": " + devices.get(i).info.name);
        }

        String devSelectProp = System.getProperty(PROP_TAD_OPENCL_DEVICES, "");
        String[] devSelectArr = devSelectProp.split(",");
        List<Device> selectedDevices = selectDevices(devices, devSelectArr);

        System.out.println("Selected devices:");
        for (Device selectedDevice : selectedDevices) {
            System.out.println("  " + selectedDevice.info.name);
        }

        return OpenCLHelper.createContext(selectedDevices);
    }

    private static List<Device> selectDevices(List<Device> devices, String[] devSelectArr) {
        if (devSelectArr.length == 1 && "".equals(devSelectArr[0])) {
            return devices;
        }
        if (devSelectArr.length > 0) {
            List<Device> r = new ArrayList<>();
            for (String deviceTag : devSelectArr) {
                r.add(findDevice(devices, deviceTag));
            }
            return r;
        }
        return devices;
    }

    private static Device findDevice(List<Device> devices, String deviceTag) {
        List<Device> byName = devices.stream()
                .filter(d -> d.info.name.contains(deviceTag))
                .collect(toList());
        if (byName.size() == 1) {
            return byName.get(0);
        } else if (byName.size() > 1) {
            throw new IllegalArgumentException("Be more specific, tag matches several devices: '" + deviceTag + "'");
        }

        try {
            return devices.get(Integer.parseInt(deviceTag));
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Device tag must be partial name or device index: '" + deviceTag + "'");
        }
    }

    @Override
    public OclArray createArray(double v) {
        return createNDArray(context, v);
    }

    @Override
    public OclArray createArray(Object multiDimArray) {
        MultiDimArrayFlattener preparedData = MultiDimArrayFlattener.prepareFlatData(multiDimArray);

        Shape shape = ProviderStore.shape(preparedData.dimensions);
        return createNDArray(context, shape, preparedData.data);
    }

    @Override
    public NDArray createArray(double[] data, Shape shape) {
        return createNDArray(context, shape, data);
    }

    @Override
    public Shape createShape(int... dims) {
        // Use the java implementation
        return new JavaShape(dims);
    }

    @Override
    public NDArray createArrayWithValue(Shape shape, double v) {
        return createNDArray(context, shape, v);
    }

}
