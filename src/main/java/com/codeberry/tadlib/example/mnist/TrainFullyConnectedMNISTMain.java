package com.codeberry.tadlib.example.mnist;

import com.codeberry.tadlib.example.mnist.SimpleTrainer.TrainParams;
import com.codeberry.tadlib.nn.model.optimizer.SGD;
import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.java.JavaProvider;
import com.codeberry.tadlib.provider.opencl.OpenCLProvider;

import static com.codeberry.tadlib.example.mnist.MNISTFullyConnectedModel.Factory.Builder.factoryBuilder;
import static com.codeberry.tadlib.example.mnist.MNISTLoader.LoadParams.params;

public class TrainFullyConnectedMNISTMain {

    public static void main(String[] args) {
//        ProviderStore.setProvider(new OpenCLProvider());
        ProviderStore.setProvider(new JavaProvider());

        SimpleTrainer trainer = new SimpleTrainer(new TrainParams()
                .batchSize(32)
                .optimizer(new SGD(0.05))
                .loaderParams(params()
                        .downloadWhenMissing(true)
                        .trainingExamples(40_000)
                        .testExamples(10_000))
                .modelFactory(factoryBuilder()
                        .hiddenNeurons(128)
                        .weightInitRandomSeed(4)
                        .build()));

        trainer.trainEpochs(5000);
    }

}
