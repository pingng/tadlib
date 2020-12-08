package com.codeberry.tadlib.nn.model.layer;

import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.tensor.Ops;
import com.codeberry.tadlib.tensor.Tensor;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Random;

import static java.util.Arrays.stream;

public interface Layer {
    Shape getOutputShape();

    default int getTotalParamValues() {
        return stream(getTrainableParams())
                .mapToInt(p -> p.getShape().size)
                .sum();
    }

    default Tensor[] getTrainableParamsNullable() {
        return new Tensor[0];
    }

    default Tensor[] getTrainableParams() {
        Tensor[] ps = getTrainableParamsNullable();
        if (ps == null) {
            return new Tensor[0];
        }
        if (hasNull(ps)) {
            return stream(ps)
                    .filter(Objects::nonNull)
                    .toArray(Tensor[]::new);
        }
        return ps;
    }

    private static boolean hasNull(Tensor[] ps) {
        for (Tensor p : ps) {
            if (p == null) {
                return true;
            }
        }
        return false;
    }

    default String getTrainingSummary() {
        return "";
    }

    ForwardResult forward(Random rnd, Tensor inputs, Ops.RunMode runMode);

    default Tensor getAdditionalCost() {
        return null;
    }

    class ForwardResult {
        public final Tensor output;
        public final Runnable[] trainingTasks;

        public ForwardResult(Tensor output) {
            this(output, null);
        }

        public ForwardResult(Tensor output, Runnable[] trainingTasks) {
            this.output = output;
            this.trainingTasks = trainingTasks;
        }

        public static ForwardResult result(Tensor output) {
            return new ForwardResult(output);
        }

        public static ForwardResult result(Tensor output, Runnable... trainingTasks) {
            return new ForwardResult(output, trainingTasks);
        }

        public void putTasksInto(List<Runnable> target) {
            if (trainingTasks != null) {
                for (Runnable tt : trainingTasks) {
                    target.add(tt);
                }
            }
        }
    }
}
