package com.codeberry.tadlib.example.mnist;

import com.codeberry.tadlib.example.mnist.SimpleMNISTTrainer.TrainParams;
import com.codeberry.tadlib.util.MultiThreadingSupport;

import static com.codeberry.tadlib.example.mnist.MNISTLoader.LoadParams.params;
import static com.codeberry.tadlib.example.mnist.MNISTConvModel.Config.Builder.cfgBuilder;

public class TrainConvMNISTMain {

    public static void main(String[] args) {
        MultiThreadingSupport.enableMultiThreading();

        SimpleMNISTTrainer trainer = new SimpleMNISTTrainer(new TrainParams()
                .batchSize(32)
                .learningRate(0.15)
                .loaderParams(params()
                        .downloadWhenMissing(true)
                        .trainingExamples(40_000)
                        .testExamples(10_000))
                .modelFactory(cfgBuilder()
                        .firstConvChannels(4)
                        .secondConvChannels(8)
                        .fullyConnectedSize(32)
                        .l2Lambda(0.01)
                        .weightInitRandomSeed(4)
                        .useBatchNormalization(true)
                        .dropoutKeep(0.5)
                        .build()));

        trainer.trainEpochs(5000);
    }
}
