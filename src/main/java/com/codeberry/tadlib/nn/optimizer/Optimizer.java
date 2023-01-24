package com.codeberry.tadlib.nn.optimizer;

import com.codeberry.tadlib.nn.optimizer.schedule.LearningRateSchedule;
import com.codeberry.tadlib.tensor.Tensor;
import com.codeberry.tadlib.util.memory.DisposalRegister;

import java.util.Collection;
import java.util.List;

import static java.util.Collections.emptyList;

public interface Optimizer {

    LearningRateSchedule learningRateSchedule();

    void optimize(List<Tensor> params);

    /**
     * @return disables that should NOT be released/freed automatically
     */
    default Collection<DisposalRegister.Disposable> getKeepInMemoryDisposables() {
        return emptyList();
    }

}
