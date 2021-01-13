package com.codeberry.tadlib.nn.model.optimizer;

import com.codeberry.tadlib.array.NDArray;
import com.codeberry.tadlib.memorymanagement.DisposalRegister;
import com.codeberry.tadlib.tensor.Tensor;

import java.util.Collection;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;

import static com.codeberry.tadlib.array.TArrayFactory.*;

public class RMSProp implements Optimizer {
    public static final double EPSILON = 1e-6;
    private final double learningRate;
    private final double gamma;

    private transient IdentityHashMap<Tensor, NDArray> sTMap = new IdentityHashMap<>();

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
                NDArray sT = updateST(p, gradient);
                NDArray sTWithEpsilon = sT.add(EPSILON);
                NDArray sqrt = sTWithEpsilon.sqrt();
                return values.sub(gradient.mul(learningRate).div(sqrt));
            });
        }
    }

    @Override
    public Collection<DisposalRegister.DisposableContainer<NDArray>> getNonDisposedContainers() {
        return Collections.unmodifiableCollection(sTMap.values());
    }

    private NDArray updateST(Tensor p, NDArray gradient) {
        NDArray sT = sTMap.get(p);
        if (sT == null) {
            sT = zeros(gradient.getShape());
        }
        NDArray gradSqr = gradient.sqr();
        sT = sT.mul(gamma).add(gradSqr.mul(1.0 - gamma));
        NDArray old = sTMap.put(p, sT);
        DisposalRegister.registerForDisposal(old);

        return sT;
    }
}
