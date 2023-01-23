package com.codeberry.tadlib.nn.model.optimizer;

import com.codeberry.tadlib.memorymanagement.DisposalRegister;
import com.codeberry.tadlib.provider.java.NDArray;
import com.codeberry.tadlib.tensor.Tensor;

import java.util.Collection;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;

import static com.codeberry.tadlib.array.TArrayFactory.*;

public class RMSProp implements Optimizer {
    public static final double EPSILON = 1.0e-6;
    private final LearningRateSchedule learningRateSchedule;
    private final double gamma;

    private final transient IdentityHashMap<Tensor, NDArray> sTMap = new IdentityHashMap<>();

    public RMSProp(LearningRateSchedule learningRateSchedule) {
        this(learningRateSchedule, 0.9);
    }

    public RMSProp(LearningRateSchedule learningRateSchedule, double gamma) {
        this.learningRateSchedule = learningRateSchedule;
        this.gamma = gamma;
    }

    @Override
    public LearningRateSchedule getLearningRateSchedule() {
        return learningRateSchedule;
    }

    @Override
    public void optimize(List<Tensor> params) {
        for (Tensor p : params) {
            p.update((values, gradient) -> {
                NDArray sT = updateST(p, gradient);
                NDArray sTWithEpsilon = sT.add(EPSILON);
                NDArray sqrt = sTWithEpsilon.sqrt();
                return values.sub(gradient.mul(learningRateSchedule.getLearningRate()).div(sqrt));
            });
        }
    }

    @Override
    public Collection<DisposalRegister.Disposable> getKeepInMemoryDisposables() {
        return Collections.unmodifiableCollection(sTMap.values());
    }

    private NDArray updateST(Tensor p, NDArray gradient) {
        NDArray sT = sTMap.get(p);
        if (sT == null) {
            sT = zeros(gradient.shape);
        }
        NDArray gradSqr = gradient.sqr();
        sT = sT.mul(gamma).add(gradSqr.mul(1.0 - gamma));
        NDArray old = sTMap.put(p, sT);
        DisposalRegister.registerForDisposal(old);

        return sT;
    }
}
