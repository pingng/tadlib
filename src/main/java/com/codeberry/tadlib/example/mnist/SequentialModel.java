package com.codeberry.tadlib.example.mnist;

import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.example.TrainingData;
import com.codeberry.tadlib.nn.model.Model;
import com.codeberry.tadlib.nn.model.ModelFactory;
import com.codeberry.tadlib.nn.model.layer.Layer;
import com.codeberry.tadlib.nn.model.layer.LayerBuilder;
import com.codeberry.tadlib.tensor.Tensor;
import com.codeberry.tadlib.util.ReflectionUtils;

import java.util.*;

import static com.codeberry.tadlib.example.mnist.MNISTLoader.*;
import static com.codeberry.tadlib.tensor.Ops.*;
import static com.codeberry.tadlib.tensor.Tensor.ZERO;
import static com.codeberry.tadlib.tensor.Tensor.constant;
import static java.util.stream.Collectors.*;

public class SequentialModel implements Model {
    private final Factory cfg;
    private final List<Layer> layers;

    public SequentialModel(Factory cfg) {
        this.cfg = cfg;

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

        int totalParams = layers.stream()
                .mapToInt(Layer::getTotalParamValues)
                .sum();
        System.out.println("Total params(doubles): " + totalParams);
    }


    private SequentialModel(SequentialModel src) {
        // init with dummy tensors for weights
        this(src.cfg);

        for (int i = 0; i < src.layers.size(); i++) {
            // overwrite weights using reflection
            ReflectionUtils.copyFieldOfClass(Tensor.class,
                    src.layers.get(i), layers.get(i),
                    Tensor::copy);
        }
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

//    public PredictionAndLosses calcGradient(Random rnd, TrainingData trainingData) {
//        resetGradients();
//
//        PredictionAndLosses l = calcCost(rnd, trainingData);
//        l.totalLoss.backward(value(1.0));
//
//        return l;
//    }

    public PredictionAndLosses calcCost(Random rnd, TrainingData trainingData) {
        int actualBatchSize = trainingData.xTrain.getShape().at(0);

        OutputWithTasks outputWithTasks = forward(rnd, trainingData.xTrain, RunMode.TRAINING);

        Tensor totalSoftmaxCost = sumSoftmaxCrossEntropy(toOneHot(trainingData.yTrain), outputWithTasks.output);
        Tensor avgSoftmaxCost = div(totalSoftmaxCost, constant(actualBatchSize));

        List<Tensor> otherCosts = layers.stream()
                .map(Layer::getAdditionalCost)
                .filter(Objects::nonNull)
                .collect(toList());

        Tensor scaledAdditionalCosts = scaleByBatch(trainingData.xTrain.getShape(), otherCosts);

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

    public Tensor predict(Tensor x_train) {
        return forward(null, x_train, RunMode.INFERENCE).output;
    }

    private OutputWithTasks forward(Random rnd, Tensor inputs, RunMode runMode) {
        List<Runnable> tasks = new ArrayList<>();

        Tensor output = inputs;
        for (Layer l : layers) {
            Layer.ForwardResult result = l.forward(rnd, output, runMode);
            result.putTasksInto(tasks);

            output = result.output;
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

    public Model copy() {
        return new SequentialModel(this);
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
