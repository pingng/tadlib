package com.codeberry.tadlib.nn.model;

import com.codeberry.tadlib.array.JavaArray;
import com.codeberry.tadlib.tensor.Tensor;

import java.util.IdentityHashMap;
import java.util.List;

import static com.codeberry.tadlib.array.TArrayFactory.*;

public class RMSProp implements Optimizer {
    public static final double EPSILON = 1e-6;
    private final double learningRate;
    private final double gamma;

    private transient IdentityHashMap<Tensor, JavaArray> sTMap = new IdentityHashMap<>();

    public RMSProp(double learningRate) {
        this(learningRate, 0.9);
    }

    public RMSProp(double learningRate, double gamma) {
        this.learningRate = learningRate;
        this.gamma = gamma;
    }

    @Override
    public void optimize(List<Tensor> params) {
        for (Tensor p : params) {
            p.update((values, gradient) -> {
                JavaArray sT = updateST(p, gradient);
                JavaArray sTWithEpsilon = sT.add(EPSILON);
                JavaArray sqrt = sTWithEpsilon.sqrt();
                return values.sub(gradient.mul(learningRate).div(sqrt));
            });
        }
    }

    private JavaArray updateST(Tensor p, JavaArray gradient) {
        JavaArray sT = sTMap.get(p);
        if (sT == null) {
            sT = zeros(gradient.shape);
        }
        JavaArray gradSqr = gradient.sqr();
        sT = sT.mul(gamma).add(gradSqr.mul(1.0 - gamma));
        sTMap.put(p, sT);

        return sT;
    }
}
