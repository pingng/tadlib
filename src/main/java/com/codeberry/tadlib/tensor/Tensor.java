package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.array.NDArray;
import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.array.TArrayFactory;
import com.codeberry.tadlib.memorymanagement.DisposalRegister;
import com.codeberry.tadlib.memorymanagement.DisposalRegister.Disposable;
import com.codeberry.tadlib.memorymanagement.DisposalRegister.DisposableContainer;
import com.codeberry.tadlib.provider.ProviderStore;

import java.util.List;
import java.util.Random;
import java.util.function.BiFunction;

import static com.codeberry.tadlib.tensor.Tensor.GradientMode.*;
import static java.util.Arrays.asList;
import static java.util.Collections.emptyList;

public class Tensor implements DisposableContainer<Disposable> {
    public static final Tensor ZERO = new Tensor(ProviderStore.array(0), NONE);

    private final List<ParentLink> links;
    private final GradientMode gradientMode;

    private NDArray vals;
    private NDArray gradient;

    public Tensor(double val) {
        this(val, CALCULATE_GRAD);
    }

    public Tensor(double val, GradientMode mode) {
        this(ProviderStore.array(val), emptyList(), mode);
    }

    public Tensor(double[][] vals) {
        this(ProviderStore.array(vals), emptyList(), CALCULATE_GRAD);
    }

    public Tensor(double[] vals) {
        this(ProviderStore.array(vals), emptyList(), CALCULATE_GRAD);
    }

    public Tensor(double[][][][] vals) {
        this(ProviderStore.array(vals), emptyList(), CALCULATE_GRAD);
    }

    public Tensor(NDArray vals) {
        this(vals, CALCULATE_GRAD);
    }

    public Tensor(NDArray vals, GradientMode gradientMode) {
        this(vals, emptyList(), gradientMode);
    }

    Tensor(NDArray vals, List<ParentLink> links) {
        this(vals, links, CALCULATE_GRAD);
    }

    private Tensor(NDArray vals, List<ParentLink> links, GradientMode gradientMode) {
        this.vals = vals;
        this.links = links;
        this.gradientMode = gradientMode;
    }

    public static Tensor tensor(double[] vals) {
        return new Tensor(vals);
    }

    public static Tensor tensor(double[][] vals) {
        return new Tensor(vals);
    }

    public static Tensor tensor(double val) {
        return new Tensor(val);
    }

    public static Tensor tensor(NDArray tArray) {
        return new Tensor(tArray);
    }

    public static Tensor constant(double val) {
        return new Tensor(val, NONE);
    }

    @Override
    public List<Disposable> getDisposables() {
        return asList(vals, gradient);
    }

    public NDArray getVals() {
        return vals;
    }

    public static abstract class TensorFactories {
        public static Tensor randomWeight(Random r, Shape shape) {
            return tensor(TArrayFactory.randomWeight(r, shape));
        }

        public static Tensor zeros(Shape shape) {
            return tensor(TArrayFactory.zeros(shape));
        }

        public static Tensor ones(Shape shape) {
            return tensor(TArrayFactory.ones(shape));
        }
    }

    public void backward() {
        backward(TArrayFactory.ones(this.vals.getShape()));
    }

    public void backward(NDArray gradient) {
        if (gradientMode == CALCULATE_GRAD) {
            if (!gradient.getShape().correspondsTo(this.vals.getShape())) {
                throw new IllegalArgumentException("Wrong shape: param:" + this.vals.getShape() + " vs grad:" + gradient.getShape());
            }

            this.gradient = this.gradient == null ?
                    gradient :
                    this.gradient.add(gradient);
//            System.out.println("Assigned gradient: "+getShape()+", "+gradient);

            for (ParentLink link : links) {
                if (link.parent.gradientMode == CALCULATE_GRAD) {
                    NDArray linkGrad = link.gradFunc.calcGradient(gradient);

                    link.parent.backward(linkGrad);
                }
            }
        }
    }

    public Object toDoubles() {
        return vals.toDoubles();
    }

    @Override
    public String toString() {
        return "Tensor{" + vals.getShape() + "}";
    }

    public Tensor subBatch(int batchId, int batchSize) {
        return new Tensor(this.vals.subBatch(batchId, batchSize), this.gradientMode);
    }

    public void update(BiFunction<NDArray, NDArray, NDArray> convertFunc) {
        testNan(vals);
        testNan(gradient);

        NDArray old = this.vals;
        this.vals = convertFunc.apply(old, this.gradient);
        testNan(this.vals);

        DisposalRegister.registerForDisposal(old);

        resetGradient();
    }

    public static void testNan(NDArray vals) {
//        if (vals != null) {
//            double[] d = vals.getInternalData();
//            for (int i = 0; i < d.length; i++) {
//                double v = d[i];
//                if (Double.isNaN(v)) {
//                    throw new RuntimeException("NAN at " + i);
//                }
//            }
//        }
    }

//    public Tensor copy() {
//        return new Tensor(vals.normalOrderedCopy(), gradientMode);
//    }

    public Shape getShape() {
        return vals.getShape();
    }

    public double[] getInternalData() {
        return vals.getInternalData();
    }

    public double dataAt(int... indices) {
        return vals.dataAt(indices);
    }

    public NDArray getGradient() {
        return gradient;
    }

    public void resetGradient() {
        if (gradient != null) {
            DisposalRegister.registerForDisposal(gradient);
        }
        this.gradient = null;
    }

    public enum GradientMode {
        NONE, CALCULATE_GRAD
    }
}
