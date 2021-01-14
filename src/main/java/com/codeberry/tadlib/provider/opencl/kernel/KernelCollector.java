package com.codeberry.tadlib.provider.opencl.kernel;

import com.codeberry.tadlib.provider.opencl.ops.*;

import java.util.ArrayList;
import java.util.List;

public class KernelCollector {
    public static List<ProgramSource> collectAllKernelSources() {
        Class<? extends OclKernelSource>[] sourceClasses = new Class[] {
                Conv.class,
                Add.class,
                Mul.class,
                Div.class,
                Sum.class,
                Simple.class,
                Clip.class,
                MaxPool.class,
                Relu.class,
                Softmax.class,
                MatMul.class,
                Transpose.class,
                Update.class
        };

        List<ProgramSource> programSources = new ArrayList<>();
        for (Class<? extends OclKernelSource> sourceClass : sourceClasses) {
            try {
                OclKernelSource src = sourceClass.getDeclaredConstructor().newInstance();
                programSources.add(new ProgramSource(sourceClass, src.getKernelSource(), src.getKernels()));
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        return programSources;
    }

    public static class ProgramSource {
        public final Class<? extends OclKernelSource> programClass;
        public final String source;
        public final List<String> kernels;

        public ProgramSource(Class<? extends OclKernelSource> programClass, String source, List<String> kernels) {
            this.programClass = programClass;
            this.source = source;
            this.kernels = kernels;
        }
    }
}
