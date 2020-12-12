package com.codeberry.tadlib.example.mnist;

import com.codeberry.tadlib.example.mnist.SimpleTrainer.TrainParams;
import com.codeberry.tadlib.nn.model.SGD;
import com.codeberry.tadlib.util.MultiThreadingSupport;

import static com.codeberry.tadlib.array.TArray.randWeight;
import static com.codeberry.tadlib.example.mnist.MNISTFullyConnectedModel.Factory.Builder.factoryBuilder;
import static com.codeberry.tadlib.example.mnist.MNISTLoader.LoadParams.params;
import static com.codeberry.tadlib.tensor.Tensor.tensor;

public class TrainFullyConnectedMNISTMain {

    public static void main(String[] args) {
        MultiThreadingSupport.enableMultiThreading();

        SimpleTrainer trainer = new SimpleTrainer(new TrainParams()
                .batchSize(32)
                .optimizer(new SGD(0.05))
                .loaderParams(params()
                        .downloadWhenMissing(true)
                        .trainingExamples(40_000)
                        .testExamples(10_000))
                .modelFactory(factoryBuilder()
                        .hiddenNeurons(32)
                        .weightInitRandomSeed(4)
                        .build()));

        trainer.trainEpochs(5000);
    }

}
