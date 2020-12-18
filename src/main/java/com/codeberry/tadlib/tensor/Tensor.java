package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.array.JavaArray;
import com.codeberry.tadlib.array.TArrayFactory;

import java.util.List;
import java.util.Random;
import java.util.function.BiFunction;

import static com.codeberry.tadlib.array.TArrayFactory.*;
import static java.util.Collections.emptyList;

public class Tensor {
    public static final Tensor ZERO = new Tensor(JavaArray.ZERO, GradientMode.NONE);

    private final List<ParentLink> links;
    private final GradientMode gradientMode;

    JavaArray vals;
    JavaArray gradient;

    public Tensor(double val) {
        this(val, GradientMode.CALCULATE_GRAD);
    }

    public Tensor(double val, GradientMode mode) {
        vals = new JavaArray(val);
        links = emptyList();
        gradientMode = mode;
    }

    public Tensor(double[][] vals) {
        this.vals = array(vals);
        links = emptyList();
        gradientMode = GradientMode.CALCULATE_GRAD;
    }

    public Tensor(double[] vals) {
        this.vals = new JavaArray(vals);
        links = emptyList();
        gradientMode = GradientMode.CALCULATE_GRAD;
    }

    public Tensor(double[][][][] vals) {
        this.vals = array(vals);
        links = emptyList();
        gradientMode = GradientMode.CALCULATE_GRAD;
    }

    public Tensor(JavaArray vals) {
        this(vals, GradientMode.CALCULATE_GRAD);
    }

    public Tensor(JavaArray vals, GradientMode gradientMode) {
        this(vals, emptyList(), gradientMode);
    }

    Tensor(JavaArray vals, List<ParentLink> links) {
        this(vals, links, GradientMode.CALCULATE_GRAD);
    }

    Tensor(JavaArray vals, List<ParentLink> links, GradientMode gradientMode) {
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

    public static Tensor tensor(JavaArray tArray) {
        return new Tensor(tArray);
    }

    public static Tensor constant(double val) {
        return new Tensor(val, GradientMode.NONE);
    }

    public static abstract class TensorFactories {
        public static Tensor randomWeight(Random r, Shape shape) {
            return tensor(randWeight(r, shape));
        }

        public static Tensor zeros(Shape shape) {
            return tensor(TArrayFactory.zeros(shape));
        }

        public static Tensor ones(Shape shape) {
            return tensor(TArrayFactory.ones(shape));
        }
    }

    public void backward() {
        backward(ones(this.vals.shape));
    }

    public void backward(JavaArray gradient) {
        if (gradientMode == GradientMode.CALCULATE_GRAD) {
            if (!gradient.shape.correspondsTo(this.vals.shape)) {
                throw new IllegalArgumentException("Wrong shape: param:" + this.vals.shape + " vs grad:" + gradient.shape);
            }

            this.gradient = this.gradient == null ?
                    gradient :
                    this.gradient.add(gradient);

            for (ParentLink link : links) {
                if (link.parent.gradientMode == GradientMode.CALCULATE_GRAD) {
                    JavaArray linkGrad = link.gradFunc.calcGradient(gradient);

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

    public void update(BiFunction<JavaArray, JavaArray, JavaArray> convertFunc) {
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

    public JavaArray getGradient() {
        return gradient;
    }

    public void resetGradient() {
        this.gradient = null;
    }

    public enum GradientMode {
        NONE, CALCULATE_GRAD
    }
}
