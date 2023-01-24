package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.array.TArrayFactory;
import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.java.NDArray;
import com.codeberry.tadlib.provider.java.Shape;
import com.codeberry.tadlib.util.memory.DisposalRegister;
import com.codeberry.tadlib.util.memory.DisposalRegister.Disposable;

import java.util.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiFunction;
import java.util.function.Consumer;

import static com.codeberry.tadlib.tensor.Tensor.GradientMode.CALCULATE_GRAD;
import static com.codeberry.tadlib.tensor.Tensor.GradientMode.NONE;
import static java.util.Arrays.asList;
import static java.util.Collections.emptyList;

public class Tensor {
    public static final Tensor ZERO = new Tensor(ProviderStore.array(0.0), NONE);
    public static final Tensor[] EMPTY_ARRAY = new Tensor[0];

    private final long id;
    private final GradLink[] links;

    /**
     * list of tensors dependent on this's value
     */
    private final Set<Tensor> linked = new LinkedHashSet<>(0);

    @Override
    public int hashCode() {
        return Long.hashCode(id);
    }

    private final GradientMode gradientMode;

    /**
     * provided with val, for setter
     */
    private final Consumer<NDArray> fwd;

    private final NDArray val;

    boolean valInvalid = true;

    private NDArray gradient;

    public Tensor(double val) {
        this(val, CALCULATE_GRAD);
    }

    public Tensor(double val, GradientMode mode) {
        this(ProviderStore.array(val), emptyList(), mode);
    }

    public Tensor(double[][] val) {
        this(ProviderStore.array(val), emptyList(), CALCULATE_GRAD);
    }

    public Tensor(double[] val) {
        this(ProviderStore.array(val), emptyList(), CALCULATE_GRAD);
    }

    public Tensor(double[][][][] val) {
        this(ProviderStore.array(val), emptyList(), CALCULATE_GRAD);
    }

    public Tensor(double[] val, int... shape) {
        this(val, Shape.shape(shape));
    }

    public Tensor(double[] val, Shape shape) {
        this(ProviderStore.array(val).reshape(shape));
    }

    public Tensor(NDArray val) {
        this(val, CALCULATE_GRAD);
    }

    public Tensor(NDArray val, GradientMode gradientMode) {
        this(val, null, emptyList(), gradientMode);
    }

    Tensor(NDArray val, List<GradLink> links) {
        this(val, null, links, CALCULATE_GRAD);
    }

    Tensor(Consumer<NDArray> fwd, Shape shape, List<GradLink> links) {
        this(new NDArray(shape), fwd, links, CALCULATE_GRAD);
    }

    private Tensor(NDArray val, List<GradLink> links, GradientMode gradientMode) {
        this(val, null, links, gradientMode);
    }

    private Tensor(NDArray val, Consumer<NDArray> fwd, List<GradLink> links, GradientMode gradientMode) {
        this.id = IdGenerator.nextId();

        this.val = val;
        this.fwd = fwd;
        this.gradientMode = gradientMode;

        this.links = links.toArray(GradLink[]::new);
        for (var l : links)
            l.parent.linked.add(this);
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
        return asList(val, gradient);
    }


    private void gradientAccum(NDArray gradient) {
        ensureGradientShape(gradient);

        this.gradient = this.gradient == null ?
                gradient :
                this.gradient.add(gradient);
    }

    @Deprecated
    private void ensureGradientShape(NDArray gradient) {
        if (!gradient.shape.correspondsTo(val.shape)) {
            throw new IllegalArgumentException("Wrong shape: param:" + this.val.shape + " vs grad:" + gradient.shape);
        }
    }

    /**
     * forward computation
     */
    public final NDArray val() {
        NDArray v = this.val;
        if (fwd != null) {
            if (valInvalid) {
                fwd.accept(v);
                valInvalid = false;
            }
        }
        return v;
    }

    public void set(NDArray x) {
        if (val.setIfDifferent(x))
            invalidateParents();
    }

    private void invalidateParents() {
        for (var l : linked) {
            l.valInvalid = true;
            l.invalidateParents();
        }
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

    private void gradientZero() {
        if (gradient != null)
            gradient.zero();

        for (var p : links)
            p.parent.gradientZero();
    }

    public void backward(boolean zero) {
        if (zero)
            gradientZero();

        backward(TArrayFactory.ones(shape()));
    }

    public void backward() {
        backward(true);
    }

    public final void backward(NDArray gradient) {
//        backwardDepthFirst(gradient);
        backwardBreadthFirst(gradient);
    }

    private void backwardDepthFirst(NDArray gradient) {
        if (gradientMode == CALCULATE_GRAD) {
            gradientAccum(gradient);

            for (var link : links)
                if (link.parent.gradientMode == CALCULATE_GRAD)
                    link.parent.backwardDepthFirst(link.gradFunc.calcGradient(gradient));
        }
    }

    private List<Tensor> backwardBreadthFirst = null;

    private void backwardBreadthFirst(NDArray gradient) {
        if (gradientMode != CALCULATE_GRAD)
            return;

        if (backwardBreadthFirst == null)
            compileBackwardBreadthFirst();

        this.gradientAccum(gradient);
        for (var next : backwardBreadthFirst) {
            NDArray g = next.gradient;
            for (var link : next.links) {
                Tensor parent = link.parent;
                if (parent.gradientMode == CALCULATE_GRAD)
                    parent.gradientAccum(link.gradFunc.calcGradient(g));
            }
        }
    }

    private void compileBackwardBreadthFirst() {
        backwardBreadthFirst = new ArrayList();

        var q = new PriorityQueue<>(IdGenerator::compareId);
        q.add(this);

        while (!q.isEmpty()) {
            Tensor next = q.poll();
            backwardBreadthFirst.add(next);

            for (var link : next.links) {
                Tensor p = link.parent;
                if (p.gradientMode == CALCULATE_GRAD)
                    q.add(p);
            }
        }
    }


    @Deprecated
    public Object toDoubles() {
        return val().toDoubles();
    }

    @Override
    public String toString() {
        NDArray ndArray = val();
        return "Tensor{" + ndArray.shape + "}";
    }

    public Tensor subBatch(int batchId, int batchSize) {
        return new Tensor(this.val().subBatch(batchId, batchSize), this.gradientMode);
    }

    public void update(BiFunction<NDArray, NDArray, NDArray> convertFunc) {
        testNaN(gradient);

        NDArray old = this.val();
        testNaN(old);

        var newVals = convertFunc.apply(old, this.gradient);
        set(newVals);

        testNaN(val);

        DisposalRegister.registerForDisposal(old);

        resetGradient();
    }

    public static void testNaN(NDArray vals) {
//        if (vals != null) {
//            double[] d = vals.getInternalData();
//            for (int i = 0; i < d.length; i++) {
//                double v = d[i];
//                if (Double.isNaN(v))
//                    throw new RuntimeException("NAN at " + i);
//            }
//        }
    }

//    public Tensor copy() {
//        return new Tensor(vals.normalOrderedCopy(), gradientMode);
//    }

    public final Shape shape() {
        return val.shape;
    }

    public double[] getInternalData() {
        return val.getInternalData();
    }

    public double dataAt(int... indices) {
        return val.dataAt(indices);
    }

    public NDArray grad() {
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
            return NEXT_ID.getAndIncrement();
        }

        static int compareId(Tensor left, Tensor right) {
            // reverse left/right since we want higher ids to be processed first
            return compareIdNaturalOrder(right.id, left.id);
        }

        static int compareIdNaturalOrder(long l, long r) {
//            long diff = diff(l, r);
//
//            if (diff > Integer.MAX_VALUE) {
//                //... interpret as wrapped, then swap values
//                long tmp = l;
//                l = r;
//                r = tmp;
//            }
            return Long.compare(l, r);
        }

//        private static long diff(long l, long r) {
//            if (l < r) {
//                return r - l;
//            }
//            return l - r;
//        }
    }
}
