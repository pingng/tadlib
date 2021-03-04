package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.array.NDArray;
import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.array.TArrayFactory;
import com.codeberry.tadlib.memorymanagement.DisposalRegister;
import com.codeberry.tadlib.memorymanagement.DisposalRegister.Disposable;
import com.codeberry.tadlib.provider.ProviderStore;

import java.util.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiFunction;

import static com.codeberry.tadlib.tensor.Tensor.GradientMode.*;
import static java.util.Arrays.asList;
import static java.util.Collections.emptyList;

public class Tensor {
    public static final Tensor ZERO = new Tensor(ProviderStore.array(0.), NONE);

    private final long id;
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
        this.id = IdGenerator.nextId();
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

    public static Tensor constant(NDArray val) {
        return new Tensor(val, NONE);
    }

    public List<Disposable> getDisposables() {
        return asList(vals, gradient);
    }

    public NDArray getVals() {
        return vals;
    }

    private void addGradient(NDArray gradient) {
        if (!gradient.getShape().correspondsTo(this.vals.getShape())) {
            throw new IllegalArgumentException("Wrong shape: param:" + this.vals.getShape() + " vs grad:" + gradient.getShape());
        }

        this.gradient = this.gradient == null ?
                gradient :
                this.gradient.add(gradient);
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
//        backwardDepthFirst(gradient);
        backwardBreadthFirst(gradient);
    }

    private void backwardDepthFirst(NDArray gradient) {
        if (gradientMode == CALCULATE_GRAD) {
            addGradient(gradient);
//            System.out.println("Assigned gradient: "+getShape()+", "+gradient);

            for (ParentLink link : links) {
                if (link.parent.gradientMode == CALCULATE_GRAD) {
                    NDArray linkGrad = link.gradFunc.calcGradient(gradient);

                    link.parent.backwardDepthFirst(linkGrad);
                }
            }
        }
    }

    private void backwardBreadthFirst(NDArray gradient) {
        if (gradientMode == CALCULATE_GRAD) {
            PriorityQueue<Tensor> gradientsToAdd = new PriorityQueue<>(IdGenerator::compareId);
            this.addGradient(gradient);
            gradientsToAdd.add(this);

            backwardBreadthFirst(gradientsToAdd);
        }
    }

    private void backwardBreadthFirst(PriorityQueue<Tensor> tensorsById) {
        while (!tensorsById.isEmpty()) {
            Tensor tensor = tensorsById.poll();
            NDArray gradient = tensor.gradient;

            for (ParentLink link : tensor.links) {
                Tensor parent = link.parent;

                if (parent.gradientMode == CALCULATE_GRAD) {
                    NDArray parentGrad = link.gradFunc.calcGradient(gradient);

                    parent.addGradient(parentGrad);

                    if (!tensorsById.contains(parent)) {
                        tensorsById.add(parent);
                    }
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

    static class IdGenerator {
        static final AtomicLong NEXT_ID = new AtomicLong();
        static long nextId() {
            return NEXT_ID.getAndUpdate(current -> {
                if (current == Long.MAX_VALUE) {
                    return 0;
                }
                return current + 1L;
            });
        }

        static int compareId(Tensor left, Tensor right) {
            // reverse left/right since we want higher ids to be processed first
            return compareIdNaturalOrder(right.id, left.id);
        }

        static int compareIdNaturalOrder(long l, long r) {
            long diff = diff(l, r);

            if (diff > Integer.MAX_VALUE) {
                //... interpret as wrapped, then swap values
                long tmp = l;
                l = r;
                r = tmp;
            }

            if (l < r) {
                return -1;
            }
            if (l > r) {
                return 1;
            }
            return 0;
        }

        private static long diff(long l, long r) {
            if (l < r) {
                return r - l;
            }
            return l - r;
        }
    }
}
