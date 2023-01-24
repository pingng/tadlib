package com.codeberry.tadlib.nn.layer;

import com.codeberry.tadlib.nn.Model;
import com.codeberry.tadlib.provider.java.Shape;
import com.codeberry.tadlib.tensor.Ops;
import com.codeberry.tadlib.tensor.Tensor;
import com.codeberry.tadlib.util.memory.DisposalRegister;

import java.util.*;

import static java.util.Arrays.stream;
import static java.util.Collections.emptyList;

public interface Layer {
    Shape getOutputShape();

    default long getTotalParamValues() {
        return stream(getTrainableParams())
                .mapToLong(p -> p.shape().size)
                .sum();
    }

    default Tensor[] getTrainableParamsNullable() {
        return Tensor.EMPTY_ARRAY;
    }

    default Tensor[] getTrainableParams() {
        Tensor[] ps = getTrainableParamsNullable();
        if (ps == null)
            return Tensor.EMPTY_ARRAY;

        if (hasNull(ps)) {
            return stream(ps)
                    .filter(Objects::nonNull)
                    .toArray(Tensor[]::new);
        }
        return ps;
    }

    private static boolean hasNull(Tensor[] ps) {
        for (Tensor p : ps) {
            if (p == null)
                return true;
        }
        return false;
    }

    default String getTrainingSummary() {
        return "";
    }

    ForwardResult forward(Random rnd, Tensor inputs, Ops.RunMode runMode, Model.IterationInfo iterationInfo);

    default Tensor getAdditionalCost() {
        return null;
    }

    /**
     * @return Objects that are needed for the model to work and thus must not be disposed
     */
    default List<? extends DisposalRegister.Disposable> getKeepInMemoryDisposables() {
        return emptyList();
    }

    class ForwardResult {
        public final Tensor output;
        public final Runnable[] trainingTasks;

        private ForwardResult(Tensor output) {
            this(output, null);
        }

        private ForwardResult(Tensor output, Runnable[] trainingTasks) {
            this.output = output;
            this.trainingTasks = trainingTasks;
        }

        public static ForwardResult result(Tensor output) {
            return new ForwardResult(output);
        }

        public static ForwardResult result(Tensor output, Runnable... trainingTasks) {
            return new ForwardResult(output, trainingTasks);
        }

        public void putTasksInto(Collection<Runnable> target) {
            if (trainingTasks != null) {
                Collections.addAll(target, trainingTasks);
            }
        }
    }
}
