package com.codeberry.tadlib.nn.model;

import com.codeberry.tadlib.array.TArray;
import com.codeberry.tadlib.tensor.Tensor;

import java.util.IdentityHashMap;
import java.util.List;

public class RMSProp implements Optimizer {
    public static final double EPSILON = 1e-6;
    private final double learningRate;
    private final double gamma;

    private transient IdentityHashMap<Tensor, TArray> sTMap = new IdentityHashMap<>();

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
                TArray sT = updateST(p, gradient);
                TArray sTWithEpsilon = sT.add(EPSILON);
                TArray sqrt = sTWithEpsilon.sqrt();
                return values.sub(gradient.mul(learningRate).div(sqrt));
            });
        }
    }

    private TArray updateST(Tensor p, TArray gradient) {
        TArray sT = sTMap.get(p);
        if (sT == null) {
            sT = TArray.zeros(gradient.shape);
        }
        TArray gradSqr = gradient.sqr();
        sT = sT.mul(gamma).add(gradSqr.mul(1.0 - gamma));
        sTMap.put(p, sT);

        return sT;
    }
}
