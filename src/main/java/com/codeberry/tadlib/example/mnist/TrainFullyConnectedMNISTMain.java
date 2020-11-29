package com.codeberry.tadlib.example.mnist;

import com.codeberry.tadlib.example.TrainingData;
import com.codeberry.tadlib.example.mnist.SimpleMNISTTrainer.TrainParams;
import com.codeberry.tadlib.tensor.Tensor;
import com.codeberry.tadlib.util.StringUtils;

import java.time.Duration;
import java.time.temporal.ChronoUnit;
import java.util.Random;

import static com.codeberry.tadlib.array.TArray.randWeight;
import static com.codeberry.tadlib.example.mnist.MNISTFullyConnectedModel.Config.Builder.cfgBuilder;
import static com.codeberry.tadlib.example.mnist.MNISTLoader.*;
import static com.codeberry.tadlib.example.mnist.MNISTLoader.LoadParams.params;
import static com.codeberry.tadlib.tensor.Tensor.tensor;
import static com.codeberry.tadlib.util.AccuracyUtils.softmaxAccuracy;

public class TrainFullyConnectedMNISTMain {

    public static void main(String[] args) {
        SimpleMNISTTrainer trainer = new SimpleMNISTTrainer(new TrainParams()
                .batchSize(32)
                .learningRate(0.05)
                .loaderParams(params()
                        .downloadWhenMissing(true)
                        .trainingExamples(40_000)
                        .testExamples(10_000))
                .modelFactory(cfgBuilder()
                        .hiddenNeurons(32)
                        .weightInitRandomSeed(4)
                        .build()));

        trainer.trainEpochs(5000);
    }

}
