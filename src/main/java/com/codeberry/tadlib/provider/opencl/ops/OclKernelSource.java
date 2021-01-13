package com.codeberry.tadlib.provider.opencl.ops;

import java.util.List;

public interface OclKernelSource {
    String getKernelSource();

    List<String> getKernels();
}
