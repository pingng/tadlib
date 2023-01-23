package com.codeberry.tadlib.example.mnist;

import com.codeberry.tadlib.nn.model.optimizer.SGD;
import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.java.JavaProvider;

import static com.codeberry.tadlib.example.mnist.MNISTLoader.LoadParams.params;
import static com.codeberry.tadlib.example.mnist.FixedMNISTConvModel.Factory.Builder.factoryBuilder;
import static com.codeberry.tadlib.example.mnist.SimpleTrainer.TrainParams.trainParams;
import static com.codeberry.tadlib.nn.model.optimizer.FixedLearningRate.*;

public class TrainFixedConvMNISTMain {

    public static void main(String[] args) {
//        ProviderStore.setProvider(new JavaProvider());
        ProviderStore.setProvider(new JavaProvider());

        SimpleTrainer trainer = new SimpleTrainer(trainParams("FixedConv")
                .batchSize(32)
                .optimizer(new SGD(fixedLearningRate(0.15)))
                .loaderParams(params()
                        .downloadWhenMissing(true)
                        .trainingExamples(40_000)
                        .testExamples(10_000))
                .modelFactory(factoryBuilder()
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
