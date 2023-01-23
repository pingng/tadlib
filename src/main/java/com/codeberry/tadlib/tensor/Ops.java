package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.array.*;
import com.codeberry.tadlib.array.util.DimensionUtils;
import com.codeberry.tadlib.nn.loss.SoftmaxCrossEntropyLoss;
import com.codeberry.tadlib.provider.java.NDArray;
import com.codeberry.tadlib.provider.java.JavaShape;
import com.codeberry.tadlib.util.MultiThreadingSupport;

import java.util.*;

import static com.codeberry.tadlib.provider.java.NDArray.*;
import static com.codeberry.tadlib.provider.java.NDArray.DimKeepRemove.KEEP_DIM;
import static com.codeberry.tadlib.provider.java.NDArray.DimKeepRemove.REMOVE_DIM;
import static com.codeberry.tadlib.array.TArrayFactory.*;
import static com.codeberry.tadlib.array.util.DimensionUtils.validateBroadcastShapes;
import static com.codeberry.tadlib.array.util.DimensionUtils.validateMatMulShapes;
import static com.codeberry.tadlib.memorymanagement.DisposalRegister.disposeAllExceptReturnedValue;
import static com.codeberry.tadlib.tensor.ParentLink.parent;
import static com.codeberry.tadlib.util.MultiThreadingSupport.TaskRange.taskRange;
import static java.lang.Boolean.TRUE;
import static java.util.Arrays.*;
import static java.util.Collections.singletonList;
import static java.util.Collections.synchronizedMap;
import static java.util.stream.Collectors.toList;

public abstract class Ops {
    public static final double EPSILON = 0.000001;

    public static Tensor MATMUL(Tensor a, Tensor b) {
        DimensionUtils.MatMulParams params = DimensionUtils.MatMulParams.expandSingleDimArrays(a.shape(), b.shape(), JavaShape::new);
        validateMatMulShapes(params.leftShape, params.rightShape);
        validateBroadcastShapes(params.leftShape, params.rightShape, -3);

        JavaShape outShape = evalMatMulShape(params.leftShape, params.rightShape);
        double[] data = new double[outShape.size];


        int outputRows = outShape.at(-2);

        JavaShape shape = (JavaShape) params.revertDimExpandOfOutputShape(outShape);

        GradFunc gF_a = grad -> disposeAllExceptReturnedValue(() ->
            aggregateBroadcastedDims(a, grad.matmul(b.val().transposeLast2D())));
        GradFunc gF_b = grad -> disposeAllExceptReturnedValue(() ->
            aggregateBroadcastedDims(b, a.val().transposeLast2D().matmul(grad)));

        MultiThreadingSupport.TaskRange totalRange = taskRange(0, outputRows).withMinimumWorkLength(decideMinimumRowsPerThread(params.leftShape, outShape));

        return new Tensor(v-> {
            NDArray left = params.promoteLeft ? new NDArray(a.val().data, (JavaShape) params.leftShape) : a.val();
            NDArray right = params.promoteRight ? new NDArray(b.val().data, (JavaShape) params.rightShape) : b.val();
            v.set(
                    MultiThreadingSupport.<double[]>multiThreadingSupportRun(
                        totalRange,
                        range -> NDArray.matmul(range, left, right, data, outShape, outShape.newIndexArray(), 0),
                        (_left, ignored_) -> _left));
            }, shape,
            asList(parent(a, gF_a), parent(b, gF_b)));
    }

    public static Tensor matmul(Tensor a, Tensor b) {
        NDArray y = a.val().matmul(b.val());

        GradFunc gF_a = grad -> disposeAllExceptReturnedValue(() -> {
            NDArray rawAGrad = grad.matmul(b.val().transposeLast2D());
            return aggregateBroadcastedDims(a, rawAGrad);
        });
        GradFunc gF_b = grad -> disposeAllExceptReturnedValue(() -> {
            NDArray rawBGrad = a.val().transposeLast2D().matmul(grad);
            return aggregateBroadcastedDims(b, rawBGrad);
        });

        return new Tensor(y,
                asList(parent(a, gF_a), parent(b, gF_b)));
    }

    public static Tensor NEGATE(Tensor x) {
        return new Tensor(y -> {
            double[] xx = x.val().data;
            double[] yy = y.data;
            for (int i = 0; i < xx.length; i++)
                yy[i] = -xx[i];
        }, x.shape(), singletonList(parent(x, NDArray::negate)));
    }

    public static Tensor negate(Tensor a) {
        NDArray y = a.val().negate();

        GradFunc gF = NDArray::negate;

        return new Tensor(y, singletonList(parent(a, gF)));
    }

    public static Tensor sub(Tensor a, Tensor b) {
        return add(a, negate(b));
    }

    public static Tensor SUB(Tensor a, Tensor b) {
        return ADD(a, NEGATE(b));
    }

    public static Tensor add(Tensor... tensors) {
        NDArray y = stream(tensors)
                .map(tensor -> tensor.val())
                .reduce(NDArray::add)
                .orElseGet(TArrayFactory::zerosShaped);

        List<ParentLink> parents = stream(tensors)
                .map(t -> parent(t, funcGradientAdd(t)))
                .collect(toList());

        return new Tensor(y, parents);
    }

    public static Tensor add(Tensor a, Tensor b) {
        NDArray y = a.val().add(b.val());

        GradFunc gF_a = funcGradientAdd(a);
        GradFunc gF_b = funcGradientAdd(b);

        return new Tensor(y, asList(parent(a, gF_a), parent(b, gF_b)));
    }
    public static Tensor ADD(Tensor a, Tensor b) {
        return new Tensor(v->{
            //TODO direct write
            v.set(a.val().add(b.val()));
        }, a.shape(), asList(parent(a, funcGradientAdd(a)), parent(b, funcGradientAdd(b))));
    }

    public static Tensor add(Tensor a, double constant) {
        NDArray y = a.val().add(constant);

        GradFunc gF_a = funcGradientAdd(a);

        return new Tensor(y, singletonList(parent(a, gF_a)));
    }

    private static GradFunc funcGradientAdd(Tensor tensor) {
        return grad -> aggregateBroadcastedDims(tensor, grad);
    }

    public static Tensor sqr(Tensor a) {
        NDArray y = a.val().sqr();

        GradFunc gF = grad -> disposeAllExceptReturnedValue(() -> grad.mul(a.val()).mul(2.0));

        return new Tensor(y, singletonList(parent(a, gF)));
    }

    public static Tensor sqrt(Tensor a) {
        NDArray y = a.val().sqrt();

        GradFunc gF = grad -> disposeAllExceptReturnedValue(() -> grad.mul(a.val().pow(-0.5).mul(0.5)));

        return new Tensor(y, singletonList(parent(a, gF)));
    }

    public static Tensor softmax(Tensor input) {
        NDArray softmax = input.val().softmax();

        // Main source: https://aerinykim.medium.com/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d
        GradFunc gF = grad -> disposeAllExceptReturnedValue(() -> {
            Shape backGradShape = grad.shape;

            NDArray separatedGrad = disposeAllExceptReturnedValue(() -> {
                NDArray selfMatMuled = disposeAllExceptReturnedValue(() -> {
                    NDArray smByColumn = softmax.reshape(softmax.shape.appendDim(1));
                    smByColumn.waitForValueReady();
                    NDArray smByRow = smByColumn.transposeLast2D();
                    smByRow.waitForValueReady();
                    return smByColumn.matmul(smByRow);
                });
                selfMatMuled.waitForValueReady();

                NDArray smDiagonal = softmax.diag();
                smDiagonal.waitForValueReady();
                return smDiagonal.sub(selfMatMuled);
            });
            separatedGrad.waitForValueReady();

            NDArray gradWithExtraDim = disposeAllExceptReturnedValue(() -> {
                NDArray gradByColumn = grad.reshape(backGradShape.appendDim(1));
                gradByColumn.waitForValueReady();
                return separatedGrad.matmul(gradByColumn);
            });
            gradWithExtraDim.waitForValueReady();

            return gradWithExtraDim.reshape(backGradShape);
        });

        return new Tensor(softmax, singletonList(parent(input, gF)));
    }

    public static Tensor div(Tensor a, Tensor b) {
        NDArray y = a.val().div(b.val());

        GradFunc gF_a = grad -> disposeAllExceptReturnedValue(() -> {
            NDArray agged = aggregateBroadcastedDims(a, grad);
            return agged.div(b.val());
        });

        GradFunc gF_b = grad -> disposeAllExceptReturnedValue(() -> {
            //  calced_divisor_grad = tf.reduce_sum(fake_grad * (-dividend / (divisor**2)), axis=(0,1))
            NDArray negateA = a.val().negate();
            NDArray rawGrad = grad.mul(negateA);
            NDArray sqrB = b.val().sqr();
            NDArray gradForB = rawGrad.div(sqrB);

            return aggregateBroadcastedDims(b, gradForB);
        });

        return new Tensor(y,
                asList(parent(a, gF_a), parent(b, gF_b)));
    }

    public static Tensor mul(Tensor a, Tensor b) {
        NDArray y = a.val().mul(b.val());
        return new Tensor(y, asList(parent(a, funcGradientMul(a, b)), parent(b, funcGradientMul(b, a))));
    }
    public static Tensor MUL(Tensor a, Tensor b) {
        validateBroadcastShapes(a.shape(), b.shape(), -1);
        JavaShape shape = evalBroadcastOutputShape(a.shape(), b.shape());

        return new Tensor(v-> {
            //TODO direct write
            v.set(a.val().mul(b.val()));
        }, shape, asList(parent(a, funcGradientMul(a, b)), parent(b, funcGradientMul(b, a))));
    }

    private static GradFunc funcGradientMul(Tensor self, Tensor other) {
        return grad -> disposeAllExceptReturnedValue(() -> {
            NDArray gr = grad.mul(other.val());

            return aggregateBroadcastedDims(self, gr);
        });
    }

    private static NDArray aggregateBroadcastedDims(Tensor self, NDArray grad) {
        return disposeAllExceptReturnedValue(() -> {
            NDArray gr = grad;
            NDArray ndArray2 = self.val();
            int missingDims = gr.shape.getDimCount() - ndArray2.shape.getDimCount();
            if (missingDims >= 1) {
                gr = gr.sumFirstDims(missingDims, REMOVE_DIM);
            }

            NDArray ndArray1 = self.val();
            if (ndArray1.shape.hasSingleDims()) {
                NDArray ndArray = self.val();
                boolean[] dimensionsToSum = ndArray.shape.forEachDim(dim -> dim == 1, boolean[]::new);
                gr = gr.sum(dimensionsToSum, KEEP_DIM);
            }
            return gr;
        });
    }

    public static Tensor sum(Tensor a) {
        NDArray y = a.val().sum();

        GradFunc gF = grad -> {
            NDArray ndArray = a.val();
            return fillLike(ndArray.shape, grad);
        };

        return new Tensor(y, singletonList(parent(a, gF)));
    }

    public static Tensor SUM(Tensor a) {
        GradFunc gF = grad -> fillLike(a.shape(), grad);

        return new Tensor(y->{
            y.set(a.val().sum());
        }, JavaShape.zeroDim, singletonList(parent(a, gF)));
    }

    public static Tensor conv2d(Tensor input, Tensor filter) {
        NDArray y = input.val().conv2d(filter.val());

        GradFunc gF_Input = grad -> disposeAllExceptReturnedValue(() -> {
            NDArray transposed = filter.val().transpose(0, 1, 3, 2);
            NDArray fT = transposed.normalOrderedCopy();
            NDArray fRotated = fT.rot180(0, 1);

            NDArray ndArray1 = filter.val();
            int offsetY = (ndArray1.shape.at(0) % 2 == 0) ? -1 : 0;
            NDArray ndArray = filter.val();
            int offsetX = (ndArray.shape.at(1) % 2 == 0) ? -1 : 0;
            return grad.conv2d(fRotated, offsetY, offsetX);
        });
        GradFunc gF_Filter = grad -> grad.calcConv2dFilterGradient(input.val(), filter.val());

        return new Tensor(y,
                asList(parent(input, gF_Input),
                        parent(filter, gF_Filter)));
    }

    public static Tensor maxpool2d(Tensor input, int size) {
        NDArray.MaxPool2dResult result = input.val().maxPool2d(size);

        GradFunc gF = grad -> grad.maxPool2dGrad(result);

        return new Tensor(result.getOutput(), singletonList(parent(input, gF)));
    }

    public static Tensor flatten(Tensor input) {
        NDArray ndArray = input.val();
        Shape inputShape = ndArray.shape;
        int size = calcFlattenExampleSize(inputShape);

        GradFunc gF = grad -> grad.reshape(inputShape);

        return new Tensor(input.val().reshape(inputShape.at(0), size),
                singletonList(parent(input, gF)));
    }

    public static int calcFlattenExampleSize(Shape inputShape) {
        int size = 1;
        for (int i = 1; i < inputShape.getDimCount(); i++) {
            size *= inputShape.at(i);
        }
        return size;
    }

    public static Tensor reshape(Tensor input, int... dims) {
        NDArray ndArray = input.val();
        Shape inputShape = ndArray.shape;

        GradFunc gF = grad -> grad.reshape(inputShape);

        return new Tensor(input.val().reshape(dims),
                singletonList(parent(input, gF)));
    }

    public static Tensor relu(Tensor input) {
        NDArray.ReluResult result = input.val().relu(0.0);
        NDArray relu = result.getOutput();

        GradFunc gF = grad -> grad.mul(result.createMask());

        return new Tensor(relu, singletonList(parent(input, gF)));
    }

    public static Tensor leakyRelu(Tensor input, double leakyScale) {
        NDArray.ReluResult result = input.val().relu(leakyScale);
        NDArray relu = result.getOutput();

        GradFunc gF = grad -> grad.mul(result.createMask());

        return new Tensor(relu, singletonList(parent(input, gF)));
    }

    public static Tensor sumSoftmaxCrossEntropy(Tensor labelsOneHot, Tensor prediction) {
        NDArray softmax = prediction.val().softmax();

        NDArray oneHotArray = labelsOneHot.val();
        NDArray cost = SoftmaxCrossEntropyLoss.sumSoftmaxCrossEntropy(softmax, oneHotArray);

        GradFunc gF = grad -> disposeAllExceptReturnedValue(() -> grad.softMaxCrossEntropyGrad(softmax, oneHotArray));

        return new Tensor(cost, singletonList(parent(prediction, gF)));
    }

    public static Tensor dropout(Tensor input, Random rnd, double dropoutKeep, RunMode runMode) {
        if (runMode == RunMode.TRAINING) {
            NDArray.DropOutResult result = input.val().dropOut(rnd, dropoutKeep);
            NDArray output = result.getOutput();

            GradFunc gF = grad -> grad.mul(result.createMask());

            return new Tensor(output, singletonList(parent(input, gF)));
        }
        return input;
    }

    public static Tensor mean(Tensor input, int... axis) {
        NDArray ndArray = input.val();
        Shape inputShape = ndArray.shape;
        boolean[] dimsToCollapse = inputShape.newCollapseArray();
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

        NDArray mean = disposeAllExceptReturnedValue(() -> {
            NDArray sum = input.val().sum(dimsToCollapse, REMOVE_DIM);
            return sum.div(finalElementsSummed);
        });

        int[] broadcastDims = inputShape.toDimArray();
        for (int d = 0; d < dimsToCollapse.length; d++) {
            if (dimsToCollapse[d]) {
                broadcastDims[d] = 1;
            }
        }

        GradFunc gF = grad -> disposeAllExceptReturnedValue(() -> {
            NDArray broadcastCompatGrad = grad.reshape(broadcastDims);
            NDArray onesInInputShape = ones(inputShape);
            NDArray gradInInputShape = onesInInputShape.mul(broadcastCompatGrad);

            return gradInInputShape.div(finalElementsSummed);
        });

        return new Tensor(mean, singletonList(parent(input, gF)));
    }

    public static Tensor sum(Tensor input, int... axis) {
        NDArray ndArray = input.val();
        Shape inputShape = ndArray.shape;
        boolean[] dimsToCollapse = inputShape.newCollapseArray();
        for (int axi : axis)
            dimsToCollapse[axi] = TRUE;

        int[] broadcastDims = inputShape.toDimArray();
        for (int d = 0; d < dimsToCollapse.length; d++) {
            if (dimsToCollapse[d]) {
                broadcastDims[d] = 1;
            }
        }

        GradFunc gF = grad -> disposeAllExceptReturnedValue(() ->
                ones(inputShape).mul(grad.reshape(broadcastDims)));

        NDArray sum = input.val().sum(dimsToCollapse, REMOVE_DIM);
        return new Tensor(sum, singletonList(parent(input, gF)));
    }

    public static Tensor SUM(Tensor x, int... axis) {
        Shape inputShape = x.shape();
        boolean[] dimsToCollapse;

        dimsToCollapse = inputShape.newCollapseArray();
        for (int axi : axis)
            dimsToCollapse[axi] = true;

        int[] broadcastDims = inputShape.toDimArray();
        for (int d = 0; d < dimsToCollapse.length; d++)
            if (dimsToCollapse[d])
                broadcastDims[d] = 1;


        //JavaShape sumShape = (JavaShape) inputShape;
        JavaShape sumShape = (JavaShape) toPhysicalShape(inputShape, dimsToCollapse);
//        if (keepRemove == DimKeepRemove.KEEP_DIM)
//            sumShape = (toPhysicalShapeWithKeep(shape, dimsToCollapse));
//        else
//            sumShape = (JavaShape) physicalShape;

        return new Tensor(y-> {
            y.zero();
            y.sum(x.val(), dimsToCollapse, REMOVE_DIM);
        }, sumShape, singletonList(parent(x,
                grad -> disposeAllExceptReturnedValue(() ->
                    ones(inputShape).mul(grad.reshape(broadcastDims))))));
    }

    public static Tensor log(Tensor x) {
        NDArray y = x.val().log();

        GradFunc gF = grad -> disposeAllExceptReturnedValue(() -> grad.div(x.val()));

        return new Tensor(y, singletonList(parent(x, gF)));
    }

    public static Tensor concat(int axis, Tensor... tensors) {
        NDArray first = tensors[0].val();
        NDArray[] rest = new NDArray[tensors.length - 1];
        for (int i = 1; i < tensors.length; i++) {
            rest[i - 1] = tensors[i].val();
        }
        NDArray concat = first.concat(rest, axis);

        int[] axisLens = Arrays.stream(tensors)
                .map(tensor1 -> tensor1.val())
                .map(ndArray -> ndArray.shape)
                .mapToInt(shape -> shape.at(axis))
                .toArray();
        GradSplitter gradSplitter = new GradSplitter(axis, axisLens);
        List<ParentLink> links = new ArrayList<>();
        for (int i = 0; i < tensors.length; i++) {
            Tensor tensor = tensors[i];
            links.add(parent(tensor, gradSplitter.getGradOfPart(i)));
        }

        return new Tensor(concat, links);
    }

    private static class GradSplitter {
        private final int axis;
        private final int[] axisLens;

        private final Map<NDArray, List<NDArray>> splitGradientsByMainGrad = synchronizedMap(new IdentityHashMap<>());

        public GradSplitter(int axis, int[] axisLens) {
            this.axis = axis;
            this.axisLens = axisLens;
        }

        GradFunc getGradOfPart(int partIndex) {
            return gradient -> {
                List<NDArray> splitGrads = ensureSplitResult(gradient);
                return splitGrads.get(partIndex);
            };
        }

        @SuppressWarnings("SynchronizationOnLocalVariableOrMethodParameter")
        private List<NDArray> ensureSplitResult(NDArray grad) {
            synchronized (grad) {
                return splitGradientsByMainGrad.computeIfAbsent(grad, gr -> gr.split(axis, axisLens));
            }
        }
    }

    public enum RunMode {
        TRAINING, INFERENCE
    }
}
