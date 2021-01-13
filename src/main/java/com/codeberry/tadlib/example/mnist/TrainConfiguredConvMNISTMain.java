package com.codeberry.tadlib.example.mnist;

import com.codeberry.tadlib.example.mnist.SimpleTrainer.TrainParams;
import com.codeberry.tadlib.nn.model.optimizer.SGD;
import com.codeberry.tadlib.nn.model.SequentialModel;
import com.codeberry.tadlib.nn.model.layer.DenseLayer;
import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.opencl.OpenCLProvider;

import static com.codeberry.tadlib.example.mnist.MNISTLoader.LoadParams.params;
import static com.codeberry.tadlib.nn.model.SequentialModel.Factory.Builder.cfgBuilder;
import static com.codeberry.tadlib.nn.model.layer.BatchNormLayer.Builder.batchNorm;
import static com.codeberry.tadlib.nn.model.layer.Conv2dLayer.BiasParam.NO_BIAS;
import static com.codeberry.tadlib.nn.model.layer.Conv2dLayer.BiasParam.USE_BIAS;
import static com.codeberry.tadlib.nn.model.layer.Conv2dLayer.Builder.conv2d;
import static com.codeberry.tadlib.nn.model.layer.DenseLayer.Builder.dense;
import static com.codeberry.tadlib.nn.model.layer.DropOutLayer.Builder.dropout;
import static com.codeberry.tadlib.nn.model.layer.MaxPool2dLayer.Builder.maxPool2d;
import static com.codeberry.tadlib.nn.model.layer.ReluLayer.Builder.relu;
import static com.codeberry.tadlib.nn.model.layer.FlattenLayer.Builder.flatten;
import static com.codeberry.tadlib.provider.ProviderStore.shape;

public class TrainConfiguredConvMNISTMain {

    public static void main(String[] args) {
//        ProviderStore.setProvider(new JavaProvider());
        ProviderStore.setProvider(new OpenCLProvider());

        SimpleTrainer trainer = new SimpleTrainer(new TrainParams()
                .batchSize(64)
//                .batchSize(32)
                .optimizer(new SGD(0.1))
//                .optimizer(new RMSProp(0.0005 * 1.5))
                .loaderParams(params()
                        .downloadWhenMissing(true)
                        .trainingExamples(40_000)
                        .testExamples(10_000))
                .modelFactory(createModelFactory(ModelSize.NORMAL_SIZE)));
//                .modelFactory(createModelFactory(ModelSize.HUGE_SIZE)));

        trainer.trainEpochs(5000);
    }

    public static SequentialModel.Factory createModelFactory(ModelSize modelSize) {
        return cfgBuilder()
                .inputShape(shape(-1, 28, 28, 1))
                .layerBuilders(
                        conv2d().biasParam(USE_BIAS).l2Lambda(0.01).filters(modelSize.filter0).kernelSize(3),
                        maxPool2d().size(2),
                        relu().leakyScale(0.01),

                        conv2d().biasParam(NO_BIAS).l2Lambda(0.01).filters(modelSize.filter1).kernelSize(3),
                        maxPool2d().size(2),
                        batchNorm(),
                        relu().leakyScale(0.01),

                        conv2d().biasParam(USE_BIAS).l2Lambda(0.01).filters(modelSize.filter2).kernelSize(3),
                        maxPool2d().size(2),
                        relu().leakyScale(0.01),

                        flatten(),
                        dense().biasParam(DenseLayer.BiasParam.NO_BIAS).l2Lambda(0.01).units(modelSize.hidden),
                        batchNorm(),
                        relu().leakyScale(0.01),

                        dropout().dropoutKeep(0.5),
                        dense().biasParam(DenseLayer.BiasParam.USE_BIAS).l2Lambda(0.01).units(10)
                )
                .weightInitRandomSeed(4)
                .build();
    }

    public enum ModelSize {
        TINY(2, 3, 4, 5),
        NORMAL_SIZE(8, 16, 16, 32),
        BIG_SIZE(8, 16, 32, 32),
        HUGE_SIZE(16, 32, 64, 64);

        final int filter0;
        final int filter1;
        final int filter2;
        final int hidden;

        ModelSize(int filter0, int filter1, int filter2, int hidden) {
            this.filter0 = filter0;
            this.filter1 = filter1;
            this.filter2 = filter2;
            this.hidden = hidden;
        }
    }
}
