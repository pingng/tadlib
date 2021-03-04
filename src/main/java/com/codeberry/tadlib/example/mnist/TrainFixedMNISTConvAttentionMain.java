package com.codeberry.tadlib.example.mnist;

import com.codeberry.tadlib.example.mnist.SimpleTrainer.TrainParams;
import com.codeberry.tadlib.nn.model.optimizer.DecayingLearningRate;
import com.codeberry.tadlib.nn.model.optimizer.FixedLearningRate;
import com.codeberry.tadlib.nn.model.optimizer.RMSProp;
import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.opencl.OpenCLProvider;

import static com.codeberry.tadlib.example.mnist.MNISTLoader.LoadParams.params;
import static com.codeberry.tadlib.example.mnist.SimpleTrainer.TrainParams.trainParams;
import static com.codeberry.tadlib.nn.model.optimizer.DecayingLearningRate.decayingLearningRate;
import static com.codeberry.tadlib.nn.model.optimizer.FixedLearningRate.fixedLearningRate;

public class TrainFixedMNISTConvAttentionMain {

    public static void main(String[] args) {
        ProviderStore.setProvider(new OpenCLProvider());

        SimpleTrainer trainer = new SimpleTrainer(trainParams("Att")
                .batchSize(32)
                // Regular MNIST
                .optimizer(new RMSProp(decayingLearningRate(0.0005, 0.0000001, 10, 0.8)))
//                .optimizer(new SGD(0.15))
                .loaderParams(params()
//                        .loadFashionMNIST()
                        .downloadWhenMissing(true)
                        .trainingExamples(40_000)
                        .testExamples(10_000))
                .modelFactory(FixedMNISTConvAttentionModel.Factory.Builder.factoryBuilder()
                        .firstConvChannels(16)
                        .secondConvChannels(32)
                        .l2Lambda(0.01)
                        .weightInitRandomSeed(4)
                        .attentionLayers(2)
                        .multiAttentionHeads(8)
                        .dropoutKeep(0.5)
                        .build()));

        trainer.trainEpochs(50000);
    }
}
