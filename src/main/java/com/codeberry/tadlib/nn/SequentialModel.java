package com.codeberry.tadlib.nn;

import com.codeberry.tadlib.nn.layer.Layer;
import com.codeberry.tadlib.nn.layer.LayerBuilder;
import com.codeberry.tadlib.provider.java.Shape;
import com.codeberry.tadlib.tensor.Tensor;
import com.codeberry.tadlib.util.Batch;
import com.codeberry.tadlib.util.TrainingDataUtils;
import com.codeberry.tadlib.util.memory.DisposalRegister;

import java.util.*;

import static com.codeberry.tadlib.tensor.Ops.*;
import static com.codeberry.tadlib.tensor.Tensor.ZERO;
import static com.codeberry.tadlib.tensor.Tensor.constant;
import static java.util.stream.Collectors.toList;

public class SequentialModel implements Model {
    //    private final Factory cfg;
    private final List<Layer> layers;

    public SequentialModel(Factory cfg) {
//        this.cfg = cfg;

        Random r = new Random(cfg.weightInitRandomSeed);


        List<Layer> layers = new ArrayList<>();
        Shape inputShape = cfg.inputShape;
        for (LayerBuilder f : cfg.layerBuilders) {
            Layer layer = f.build(r, inputShape);
            Shape outputShape = layer.getOutputShape();

            System.out.println("Input: " + inputShape + " output: " + outputShape + " params: " + layer.getTotalParamValues() + " layer: " + layer.getClass().getSimpleName());
            inputShape = outputShape;

            layers.add(layer);
        }
        this.layers = layers;

        long totalParams = layers.stream()
                .mapToLong(Layer::getTotalParamValues)
                .sum();
        System.out.println("Total params(doubles): " + totalParams);
    }

    @Override
    public String getTrainingLogText() {
        StringBuilder buf = new StringBuilder();
        for (Layer l : layers) {
            String summary = l.getTrainingSummary();
            if (!summary.isBlank()) {
                buf.append(summary).append("\n");
            }
        }
        return buf.toString();
    }

    public PredictionAndLosses calcCost(Random rnd, Batch trainingData, IterationInfo iterationInfo) {
        int actualBatchSize = trainingData.getBatchSize();

        OutputWithTasks outputWithTasks = forward(rnd, trainingData.input, RunMode.TRAINING, iterationInfo);

        Tensor totalSoftmaxCost = sumSoftmaxCrossEntropy(TrainingDataUtils.toOneHot(trainingData.output, 10), outputWithTasks.output);
        Tensor avgSoftmaxCost = div(totalSoftmaxCost, constant(actualBatchSize));

        List<Tensor> otherCosts = layers.stream()
                .map(Layer::getAdditionalCost)
                .filter(Objects::nonNull)
                .collect(toList());

        Tensor scaledAdditionalCosts = scaleByBatch(trainingData.input.shape(), otherCosts);

        Tensor totalLoss = add(avgSoftmaxCost, scaledAdditionalCosts);

        return new PredictionAndLosses(outputWithTasks.output,
                outputWithTasks.trainingTasks,
                totalLoss, scaledAdditionalCosts);
    }

    private static Tensor scaleByBatch(Shape inputShape, List<Tensor> otherCosts) {
        if (!otherCosts.isEmpty()) {
            Tensor sum = add(otherCosts.toArray(Tensor[]::new));

            int actualBatchSize = inputShape.at(0);
            return mul(sum, constant(1.0 / actualBatchSize));
        }
        return ZERO;
    }

    public List<Tensor> getParams() {
        return layers.stream()
                .map(Layer::getTrainableParams)
                .flatMap(Arrays::stream)
                .filter(Objects::nonNull)
                .collect(toList());
    }

    public List<DisposalRegister.Disposable> getKeepInMemoryDisposables() {
        return layers.stream()
                .map(Layer::getKeepInMemoryDisposables)
                .flatMap(List::stream)
                .filter(Objects::nonNull)
                .collect(toList());
    }

    public Tensor predict(Tensor x_train, IterationInfo iterationInfo) {
        return forward(null, x_train, RunMode.INFERENCE, iterationInfo).output;
    }

    private OutputWithTasks forward(Random rnd, Tensor inputs, RunMode runMode, IterationInfo iterationInfo) {
        List<Runnable> tasks = new ArrayList<>();

        Tensor output = inputs;
        for (Layer l : layers) {
//            double[] d = output.getInternalData();
//            for (double v : d) {
//                if (Double.isNaN(v)) {
//                    throw new RuntimeException("NAN!");
//                }
//            }

            Layer.ForwardResult result = l.forward(rnd, output, runMode, iterationInfo);
            result.putTasksInto(tasks);

            output = result.output;
//            d = output.getInternalData();
//            for (double v : d) {
//                if (Double.isNaN(v)) {
//                    throw new RuntimeException("NAN!");
//                }
//            }
        }

        return new OutputWithTasks(output, tasks);
    }

    private static class OutputWithTasks {
        final Tensor output;
        final List<Runnable> trainingTasks;

        private OutputWithTasks(Tensor output, List<Runnable> trainingTasks) {
            this.output = output;
            this.trainingTasks = trainingTasks;
        }
    }

    public static class Factory implements ModelFactory {
        private final Shape inputShape;
        private final LayerBuilder[] layerBuilders;
        private final long weightInitRandomSeed;

        public Factory(Shape inputShape, LayerBuilder[] layerBuilders, long weightInitRandomSeed) {
            this.inputShape = inputShape;
            this.layerBuilders = layerBuilders;
            this.weightInitRandomSeed = weightInitRandomSeed;
        }

        @Override
        public Model createModel() {
            return new SequentialModel(this);
        }

        public static class Builder {
            private Shape inputShape;
            private LayerBuilder[] layerBuilders;
            private long weightInitRandomSeed = 4;

            public static Builder cfgBuilder() {
                return new Builder();
            }

            public Builder inputShape(Shape inputShape) {
                this.inputShape = inputShape;
                return this;
            }

            public Builder layerBuilders(LayerBuilder... layerBuilders) {
                this.layerBuilders = layerBuilders;
                return this;
            }

            public Builder weightInitRandomSeed(long weightInitRandomSeed) {
                this.weightInitRandomSeed = weightInitRandomSeed;
                return this;
            }

            public Factory build() {
                return new Factory(inputShape, layerBuilders, weightInitRandomSeed);
            }
        }
    }
}
