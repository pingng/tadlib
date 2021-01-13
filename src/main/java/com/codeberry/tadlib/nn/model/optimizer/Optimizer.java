package com.codeberry.tadlib.nn.model.optimizer;

import com.codeberry.tadlib.array.NDArray;
import com.codeberry.tadlib.memorymanagement.DisposalRegister;
import com.codeberry.tadlib.tensor.Tensor;

import java.util.Collection;
import java.util.List;

import static java.util.Collections.emptyList;

public interface Optimizer {
    void optimize(List<Tensor> params);

    default Collection<DisposalRegister.DisposableContainer<NDArray>> getNonDisposedContainers() {
        return emptyList();
    }

}
