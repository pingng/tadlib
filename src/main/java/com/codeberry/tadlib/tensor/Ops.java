package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.array.*;
import com.codeberry.tadlib.nn.loss.SoftmaxCrossEntropyLoss;
import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.java.JavaArray;

import java.util.List;
import java.util.Random;

import static com.codeberry.tadlib.array.NDArray.*;
import static com.codeberry.tadlib.array.NDArray.DimKeepRemove.KEEP_DIM;
import static com.codeberry.tadlib.array.NDArray.DimKeepRemove.REMOVE_DIM;
import static com.codeberry.tadlib.array.TArrayFactory.*;
import static com.codeberry.tadlib.memorymanagement.DisposalRegister.*;
import static com.codeberry.tadlib.tensor.ParentLink.parentLink;
import static com.codeberry.tadlib.util.MultiThreadingSupport.TaskRange.taskRange;
import static java.lang.Boolean.TRUE;
import static java.util.Arrays.*;
import static java.util.Collections.singletonList;
import static java.util.stream.Collectors.toList;

public abstract class Ops {
    public static final double EPSILON = 0.000001;

    public static Tensor matmul(Tensor a, Tensor b) {
        NDArray y = a.getVals().matmul(b.getVals());

        GradFunc gF_a = grad -> disposeAllExceptReturnedValues(() -> grad.matmul(b.getVals().transpose()));
        GradFunc gF_b = grad -> disposeAllExceptReturnedValues(() -> a.getVals().transpose().matmul(grad));

        return new Tensor(y,
                asList(parentLink(a, gF_a), parentLink(b, gF_b)));
    }

    public static Tensor negate(Tensor a) {
        NDArray y = a.getVals().negate();

        GradFunc gF = NDArray::negate;

        return new Tensor(y,
                singletonList(parentLink(a, gF)));
    }

    public static Tensor sub(Tensor a, Tensor b) {
        Tensor negated = negate(b);
        return add(a, negated);
    }

    public static Tensor add(Tensor... tensors) {
        NDArray y = stream(tensors)
                .map(t -> t.getVals())
                .reduce(NDArray::add)
                .orElse(JavaArray.ZERO);

        List<ParentLink> parents = stream(tensors)
                .map(t -> parentLink(t, funcGradientAdd(t)))
                .collect(toList());

        return new Tensor(y, parents);
    }

    public static Tensor add(Tensor a, Tensor b) {
        NDArray y = a.getVals().add(b.getVals());

        GradFunc gF_a = funcGradientAdd(a);
        GradFunc gF_b = funcGradientAdd(b);

        return new Tensor(y,
                asList(parentLink(a, gF_a), parentLink(b, gF_b)));
    }

    public static Tensor add(Tensor a, double constant) {
        NDArray y = a.getVals().add(constant);

        GradFunc gF_a = funcGradientAdd(a);

        return new Tensor(y, singletonList(parentLink(a, gF_a)));
    }

    private static GradFunc funcGradientAdd(Tensor tensor) {
        return grad -> aggregateBroadcastedDims(tensor, grad);
    }

    public static Tensor sqr(Tensor a) {
        NDArray y = a.getVals().sqr();

        GradFunc gF = grad -> disposeAllExceptReturnedValues(() -> grad.mul(a.getVals()).mul(2.0));

        return new Tensor(y, singletonList(parentLink(a, gF)));
    }

    public static Tensor sqrt(Tensor a) {
        NDArray y = a.getVals().sqrt();

        GradFunc gF = grad -> disposeAllExceptReturnedValues(() -> grad.mul(a.getVals().pow(-0.5).mul(0.5)));

        return new Tensor(y, singletonList(parentLink(a, gF)));
    }

    public static Tensor div(Tensor a, Tensor b) {
        NDArray y = a.getVals().div(b.getVals());

        GradFunc gF_a = grad -> disposeAllExceptReturnedValues(()-> {
            NDArray agged = aggregateBroadcastedDims(a, grad);
            return agged.div(b.getVals());
        });

        GradFunc gF_b = grad -> disposeAllExceptReturnedValues(() -> {
            //  calced_divisor_grad = tf.reduce_sum(fake_grad * (-dividend / (divisor**2)), axis=(0,1))
            NDArray negateA = a.getVals().negate();
            NDArray rawGrad = grad.mul(negateA);
            NDArray sqrB = b.getVals().sqr();
            NDArray gradForB = rawGrad.div(sqrB);

            return aggregateBroadcastedDims(b, gradForB);
        });

        return new Tensor(y,
                asList(parentLink(a, gF_a), parentLink(b, gF_b)));
    }

    public static Tensor mul(Tensor a, Tensor b) {
        NDArray y = a.getVals().mul(b.getVals());

        GradFunc gF_a = funcGradientMul(a, b);
        GradFunc gF_b = funcGradientMul(b, a);

        return new Tensor(y,
                asList(parentLink(a, gF_a), parentLink(b, gF_b)));
    }

    private static GradFunc funcGradientMul(Tensor self, Tensor other) {
        return grad -> disposeAllExceptReturnedValues(() -> {
            NDArray gr = grad.mul(other.getVals());

            return aggregateBroadcastedDims(self, gr);
        });
    }

    private static NDArray aggregateBroadcastedDims(Tensor self, NDArray grad) {
        return disposeAllExceptReturnedValues(() -> {
            NDArray gr = grad;
            int missingDims = gr.getShape().getDimCount() - self.getVals().getShape().getDimCount();
            if (missingDims >= 1) {
                gr = gr.sumFirstDims(missingDims, REMOVE_DIM);
            }

            if (self.getVals().getShape().hasSingleDims()) {
                Boolean[] dimensionsToSum = self.getVals().getShape().forEachDim(dim -> dim == 1, Boolean[]::new);
                gr = gr.sum(dimensionsToSum, KEEP_DIM);
            }
            return gr;
        });
    }

    public static Tensor sum(Tensor a) {
        NDArray y = a.getVals().sum();

        GradFunc gF = grad -> fillLike(a.getVals().getShape(), grad);

        return new Tensor(y,
                singletonList(parentLink(a, gF)));
    }

    public static Tensor conv2d(Tensor input, Tensor filter) {
        NDArray y = input.getVals().conv2d(filter.getVals());

        GradFunc gF_Input = grad -> disposeAllExceptReturnedValues(() -> {
            NDArray transposed = filter.getVals().transpose(0, 1, 3, 2);
            NDArray fT = transposed.normalOrderedCopy();
            NDArray fRotated = fT.rot180(0, 1);

            int offsetY = (filter.getVals().getShape().at(0) % 2 == 0) ? -1 : 0;
            int offsetX = (filter.getVals().getShape().at(1) % 2 == 0) ? -1 : 0;
            return grad.conv2d(fRotated, offsetY, offsetX);
        });
        GradFunc gF_Filter = grad -> grad.calcConv2dFilterGradient(input.getVals(), filter.getVals());

        return new Tensor(y,
                asList(parentLink(input, gF_Input),
                        parentLink(filter, gF_Filter)));
    }

    public static Tensor maxpool2d(Tensor input, int size) {
        MaxPool2dResult result = input.getVals().maxPool2d(size);

        GradFunc gF = grad -> grad.maxPool2dGrad(result);

        return new Tensor(result.getOutput(), singletonList(parentLink(input, gF)));
    }

    public static Tensor flatten(Tensor input) {
        Shape inputShape = input.getVals().getShape();
        int size = calcFlattenExampleSize(inputShape);

        GradFunc gF = grad -> grad.reshape(inputShape);

        return new Tensor(input.getVals().reshape(inputShape.at(0), size),
                singletonList(parentLink(input, gF)));
    }

    public static int calcFlattenExampleSize(Shape inputShape) {
        int size = 1;
        for (int i = 1; i < inputShape.getDimCount(); i++) {
            size *= inputShape.at(i);
        }
        return size;
    }

    public static Tensor reshape(Tensor input, int... dims) {
        Shape inputShape = input.getVals().getShape();

        GradFunc gF = grad -> grad.reshape(inputShape);

        return new Tensor(input.getVals().reshape(dims),
                singletonList(parentLink(input, gF)));
    }

    public static Tensor relu(Tensor input) {
        ReluResult result = input.getVals().relu(0.0);
        NDArray relu = result.getOutput();

        GradFunc gF = grad -> grad.mul(result.createMask());

        return new Tensor(relu, singletonList(parentLink(input, gF)));
    }

    public static Tensor leakyRelu(Tensor input, double leakyScale) {
        ReluResult result = input.getVals().relu(leakyScale);
        NDArray relu = result.getOutput();

        GradFunc gF = grad -> grad.mul(result.createMask());

        return new Tensor(relu, singletonList(parentLink(input, gF)));
    }

    public static Tensor sumSoftmaxCrossEntropy(Tensor labelsOneHot, Tensor prediction) {
        NDArray softmax = prediction.getVals().softmax();

        NDArray oneHotArray = labelsOneHot.getVals();
        NDArray cost = SoftmaxCrossEntropyLoss.sumSoftmaxCrossEntropy(softmax, oneHotArray);

        GradFunc gF = grad -> {
//            TMutableArray tgt = TMutableArray.copyOf(softmax);
//            toSoftmaxGradient(tgt, softmax, softmax.getShape().newIndexArray(),
//                    oneHotArray, 0);
//            return tgt.migrateToImmutable().mul(grad);
            return grad.softMaxGrad(softmax, oneHotArray);
        };

        return new Tensor(cost, singletonList(parentLink(prediction, gF)));
    }

    public static Tensor dropout(Tensor input, Random rnd, double dropoutKeep, RunMode runMode) {
        if (runMode == RunMode.TRAINING) {
            DropOutResult result = input.getVals().dropOut(rnd, dropoutKeep);
            NDArray output = result.getOutput();

            GradFunc gF = grad -> grad.mul(result.createMask());

            return new Tensor(output, singletonList(parentLink(input, gF)));
        }
        return input;
    }

    public static Tensor mean(Tensor input, int... axis) {
        Shape inputShape = input.getVals().getShape();
        Boolean[] dimsToCollapse = inputShape.newCollapseArray();
        for (int axi : axis) {
            dimsToCollapse[axi] = TRUE;
        }
        int elementsSummed = 1;
        for (int i = 0; i < inputShape.getDimCount(); i++) {
            if (dimsToCollapse[i]) {
                elementsSummed *= inputShape.at(i);
            }
        }
        int finalElementsSummed = elementsSummed;

        NDArray mean = disposeAllExceptReturnedValues(() -> {
            NDArray sum = input.getVals().sum(dimsToCollapse, REMOVE_DIM);
            return sum.div(finalElementsSummed);
        });

        int[] broadcastDims = inputShape.toDimArray();
        for (int d = 0; d < dimsToCollapse.length; d++) {
            if (dimsToCollapse[d]) {
                broadcastDims[d] = 1;
            }
        }

        GradFunc gF = grad -> disposeAllExceptReturnedValues(() -> {
            NDArray broadcastCompatGrad = grad.reshape(broadcastDims);
            NDArray onesInInputShape = ones(inputShape);
            NDArray gradInInputShape = onesInInputShape.mul(broadcastCompatGrad);

            return gradInInputShape.div(finalElementsSummed);
        });

        return new Tensor(mean, singletonList(parentLink(input, gF)));
    }

    public static Tensor log(Tensor input) {
        NDArray y = input.getVals().log();

        GradFunc gF = grad -> disposeAllExceptReturnedValues(() -> grad.div(input.getVals()));

        return new Tensor(y, singletonList(parentLink(input, gF)));
    }

    public enum RunMode {
        TRAINING, INFERENCE
    }
}
