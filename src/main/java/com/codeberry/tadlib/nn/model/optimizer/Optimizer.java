package com.codeberry.tadlib.nn.model.optimizer;

import com.codeberry.tadlib.memorymanagement.DisposalRegister;
import com.codeberry.tadlib.tensor.Tensor;

import java.util.Collection;
import java.util.List;

import static java.util.Collections.emptyList;

public interface Optimizer {

    LearningRateSchedule getLearningRateSchedule();

    void optimize(List<Tensor> params);

    /**
     * @return disables that should NOT be released/freed automatically
     */
    default Collection<DisposalRegister.Disposable> getKeepInMemoryDisposables() {
        return emptyList();
    }

}
