package com.codeberry.tadlib.example.mnist;

import com.codeberry.tadlib.example.mnist.SimpleTrainer.TrainParams;
import com.codeberry.tadlib.nn.model.optimizer.SGD;
import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.opencl.OpenCLProvider;

import static com.codeberry.tadlib.example.mnist.MNISTLoader.LoadParams.params;
import static com.codeberry.tadlib.example.mnist.FixedMNISTConvModel.Factory.Builder.factoryBuilder;

public class TrainFixedConvMNISTMain {

    public static void main(String[] args) {
        ProviderStore.setProvider(new OpenCLProvider());

        SimpleTrainer trainer = new SimpleTrainer(new TrainParams()
                .batchSize(32)
                .optimizer(new SGD(0.15))
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
