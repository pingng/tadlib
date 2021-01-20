package com.codeberry.tadlib.provider.opencl.ops;

import com.codeberry.tadlib.array.Comparison;
import com.codeberry.tadlib.array.NDArray;
import com.codeberry.tadlib.array.NDIntArray;
import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.opencl.*;
import com.codeberry.tadlib.provider.opencl.context.Context;
import com.codeberry.tadlib.provider.opencl.kernel.Kernel;
import com.codeberry.tadlib.provider.opencl.queue.CommandQueue;

import java.util.Collections;
import java.util.List;

import static com.codeberry.tadlib.array.util.DimensionUtils.*;
import static com.codeberry.tadlib.provider.opencl.OclArray.createNDArray;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.ByteSize.sizeOf;
import static com.codeberry.tadlib.provider.opencl.OclBuffer.createBuffer;
import static com.codeberry.tadlib.provider.opencl.OclDataType.cl_double;
import static com.codeberry.tadlib.provider.opencl.OclDataType.cl_int;
import static com.codeberry.tadlib.provider.opencl.buffer.BufferMemFlags.CL_MEM_READ_WRITE;
import static com.codeberry.tadlib.provider.opencl.buffer.BufferMemFlags.valueOf;
import static com.codeberry.tadlib.provider.opencl.kernel.Kernel.*;
import static com.codeberry.tadlib.provider.opencl.queue.CommandQueue.WorkItemMode.MULTIPLES_OF_PREFERRED_GROUP_SIZE;
import static com.codeberry.tadlib.util.ClassResourceUtils.readString;

public class Compare implements OclKernelSource {

    public static final String COMPARE = "compare";

    @Override
    public String getKernelSource() {
        return readString(Compare.class, "Compare.cl");
    }

    @Override
    public List<String> getKernels() {
        return Collections.singletonList(COMPARE);
    }

    public static NDArray compare(Context context, OclArray left, OclArray right, Comparison comparison, double trueValue, double falseValue) {
        return compare_(context, new ArrayType(left), new ArrayType(right),
                comparison,
                new DoubleValue(trueValue), new DoubleValue(falseValue),
                left.getShape(), right.getShape(),
                cl_double, OclArray::createNDArray);
    }

    public static NDArray compare(Context context, OclArray left, OclIntArray right, Comparison comparison, double trueValue, double falseValue) {
        return compare_(context, new ArrayType(left), new ArrayType(right),
                comparison,
                new DoubleValue(trueValue), new DoubleValue(falseValue),
                left.getShape(), right.getShape(),
                cl_double, OclArray::createNDArray);
    }

    public static NDArray compare(Context context, OclIntArray left, OclArray right, Comparison comparison, double trueValue, double falseValue) {
        return compare_(context, new ArrayType(left), new ArrayType(right),
                comparison,
                new DoubleValue(trueValue), new DoubleValue(falseValue),
                left.getShape(), right.getShape(),
                cl_double, OclArray::createNDArray);
    }

    public static NDIntArray compare(Context context, OclIntArray left, OclIntArray right, Comparison comparison, int trueValue, int falseValue) {
        return compare_(context, new ArrayType(left), new ArrayType(right),
                comparison,
                new IntValue(trueValue), new IntValue(falseValue),
                left.getShape(), right.getShape(),
                cl_int, OclIntArray::createNDArray);
    }

    private static <R, V> R compare_(Context context, ArrayType leftType, ArrayType rightType,
                                  Comparison comparison,
                                  OutputValue<V> trueValue, OutputValue<V> falseValue,
                                  Shape leftShape, Shape rightShape,
                                  OclDataType outDataType,
                                  ArrayFactory<R> factory) {
        validateBroadcastShapes(leftShape, rightShape, -1);
        Shape outShape = ProviderStore.shape(createBroadcastResultDims(leftShape, rightShape));
        int outDimCount = outShape.getDimCount();

        OclBuffer out = createBuffer(context, sizeOf(outDataType, outShape.getSize()), CL_MEM_READ_WRITE);

        CommandQueue queue = context.getQueue();
        Kernel kernel = context.findKernel(COMPARE);
        InProgressResources resources = new InProgressResources(context);

        int comparisonCode = ComparisonCode.codeOf(comparison);

        ArgSetter argSetter = kernel.createArgSetter(resources);
        leftType.assignArrayAndType(argSetter);
        rightType.assignArrayAndType(argSetter);
        trueValue.assignValue(argSetter);
        falseValue.assignValue(argSetter);
        argSetter
                .nextArg(calcBroadcastBlockSizes(leftShape, outDimCount))
                .nextArg(calcBroadcastBlockSizes(rightShape, outDimCount))
                .nextArg(comparisonCode)
                .nextArg(comparison.getDelta())
                .nextArgKeepRef(out)
                .nextArg(outDimCount)
                .nextArg(calcBlockSizes(outShape))
                .nextArg(outShape.getSize());

        queue.enqueueKernel(kernel, outShape.getSize(),
                MULTIPLES_OF_PREFERRED_GROUP_SIZE, resources);

        return factory.createArray(outShape, out, resources);
    }

    private interface ArrayFactory<R> {
        R createArray(Shape outShape, OclBuffer buffer, InProgressResources resources);
    }

    private interface OutputValue<V> {
        void assignValue(ArgSetter argSetter);
    }

    private static class DoubleValue implements OutputValue<Double> {
        private final double value;

        DoubleValue(double value) {
            this.value = value;
        }

        @Override
        public void assignValue(ArgSetter argSetter) {
            argSetter.nextArg(new double[]{value});
        }
    }

    private static class IntValue implements OutputValue<Integer> {
        private final int value;

        IntValue(int value) {
            this.value = value;
        }

        @Override
        public void assignValue(ArgSetter argSetter) {
            argSetter.nextArg(new int[]{value});
        }
    }

    private static class ArrayType {
        static final int DOUBLE = 0;
        static final int INT = 1;

        private final OclArray doubleArray;
        private final OclIntArray intArray;

        ArrayType(OclArray array) {
            intArray = null;
            doubleArray = array;
        }

        public ArrayType(OclIntArray intArray) {
            this.intArray = intArray;
            this.doubleArray = null;
        }

        public void assignArrayAndType(ArgSetter argSetter) {
            if (doubleArray != null) {
                argSetter.nextArg(doubleArray)
                        .nextArg(DOUBLE);
            } else {
                //...then i sint array
                argSetter.nextArg(intArray)
                        .nextArg(INT);
            }
        }
    }

    private static abstract class ComparisonCode {
        private static final int CMP_EQUALS = 0;
        private static final int CMP_GREATER_EQUALS = 1;
        private static final int CMP_LESS_EQUALS = 2;
        private static final int CMP_GREATER = 3;
        private static final int CMP_LESS = 4;

        private static int codeOf(Comparison comparison) {
            switch (comparison.getId()) {
                case "==":
                    return CMP_EQUALS;

                case ">=":
                    return CMP_GREATER_EQUALS;

                case "<=":
                    return CMP_LESS_EQUALS;

                case ">":
                    return CMP_GREATER;

                case "<":
                    return CMP_LESS;
            }

            throw new IllegalArgumentException("Unknown " + Comparison.class.getSimpleName() + ": " + comparison);
        }
    }
}
