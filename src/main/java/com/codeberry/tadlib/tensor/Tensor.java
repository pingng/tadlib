package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.array.TArray;

import java.util.List;
import java.util.Random;
import java.util.function.BiFunction;

import static java.util.Collections.emptyList;

public class Tensor {
    public static final Tensor ZERO = new Tensor(TArray.ZERO, GradientMode.NONE);

    private final List<ParentLink> links;
    private final GradientMode gradientMode;

    TArray vals;
    TArray gradient;

    public Tensor(double val) {
        this(val, GradientMode.CALCULATE_GRAD);
    }

    public Tensor(double val, GradientMode mode) {
        vals = new TArray(val);
        links = emptyList();
        gradientMode = mode;
    }

    public Tensor(double[][] vals) {
        this.vals = new TArray(vals);
        links = emptyList();
        gradientMode = GradientMode.CALCULATE_GRAD;
    }

    public Tensor(double[] vals) {
        this.vals = new TArray(vals);
        links = emptyList();
        gradientMode = GradientMode.CALCULATE_GRAD;
    }

    public Tensor(double[][][][] vals) {
        this.vals = new TArray(vals);
        links = emptyList();
        gradientMode = GradientMode.CALCULATE_GRAD;
    }

    public Tensor(TArray vals) {
        this(vals, GradientMode.CALCULATE_GRAD);
    }

    public Tensor(TArray vals, GradientMode gradientMode) {
        this(vals, emptyList(), gradientMode);
    }

    Tensor(TArray vals, List<ParentLink> links) {
        this(vals, links, GradientMode.CALCULATE_GRAD);
    }

    Tensor(TArray vals, List<ParentLink> links, GradientMode gradientMode) {
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

    public static Tensor tensor(TArray tArray) {
        return new Tensor(tArray);
    }

    public static Tensor constant(double val) {
        return new Tensor(val, GradientMode.NONE);
    }

    public static abstract class TensorFactories {
        public static Tensor randomWeight(Random r, Shape shape) {
            return tensor(TArray.randWeight(r, shape));
        }

        public static Tensor zeros(Shape shape) {
            return tensor(TArray.zeros(shape));
        }

        public static Tensor ones(Shape shape) {
            return tensor(TArray.ones(shape));
        }
    }

    public void backward() {
        backward(TArray.ones(this.vals.shape));
    }

    public void backward(TArray gradient) {
        if (gradientMode == GradientMode.CALCULATE_GRAD) {
            if (!gradient.shape.correspondsTo(this.vals.shape)) {
                throw new IllegalArgumentException("Wrong shape: param:" + this.vals.shape + " vs grad:" + gradient.shape);
            }

            this.gradient = this.gradient == null ?
                    gradient :
                    this.gradient.add(gradient);

            for (ParentLink link : links) {
                if (link.parent.gradientMode == GradientMode.CALCULATE_GRAD) {
                    TArray linkGrad = link.gradFunc.calcGradient(gradient);

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
        return "Tensor{" + vals.shape + "}";
    }

    public Tensor subBatch(int batchId, int batchSize) {
        return new Tensor(this.vals.subBatch(batchId, batchSize), this.gradientMode);
    }

    public void update(BiFunction<TArray, TArray, TArray> convertFunc) {
        this.vals = convertFunc.apply(this.vals, this.gradient);

        resetGradient();
    }

    public Tensor copy() {
        return new Tensor(vals.normalOrderedCopy(), gradientMode);
    }

    public Shape getShape() {
        return vals.shape;
    }

    public double[] getInternalData() {
        return vals.getInternalData();
    }

    public double dataAt(int... indices) {
        return vals.dataAt(indices);
    }

    public TArray getGradient() {
        return gradient;
    }

    public void resetGradient() {
        this.gradient = null;
    }

    public enum GradientMode {
        NONE, CALCULATE_GRAD
    }
}
