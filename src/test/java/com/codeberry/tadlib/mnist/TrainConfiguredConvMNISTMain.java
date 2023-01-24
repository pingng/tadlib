package com.codeberry.tadlib.mnist;

import com.codeberry.tadlib.nn.SequentialModel;
import com.codeberry.tadlib.nn.layer.DenseLayer;

import static com.codeberry.tadlib.nn.SequentialModel.Factory.Builder.cfgBuilder;
import static com.codeberry.tadlib.nn.layer.BatchNormLayer.Builder.batchNorm;
import static com.codeberry.tadlib.nn.layer.BlockDropOutLayer.Builder.blockDropout;
import static com.codeberry.tadlib.nn.layer.Conv2dLayer.BiasParam.NO_BIAS;
import static com.codeberry.tadlib.nn.layer.Conv2dLayer.BiasParam.USE_BIAS;
import static com.codeberry.tadlib.nn.layer.Conv2dLayer.Builder.conv2d;
import static com.codeberry.tadlib.nn.layer.DenseLayer.Builder.dense;
import static com.codeberry.tadlib.nn.layer.DropOutLayer.Builder.dropout;
import static com.codeberry.tadlib.nn.layer.FlattenLayer.Builder.flatten;
import static com.codeberry.tadlib.nn.layer.MaxPool2dLayer.Builder.maxPool2d;
import static com.codeberry.tadlib.nn.layer.ProportionalDropOutLayer.Builder.proportionalDropout;
import static com.codeberry.tadlib.nn.layer.ReluLayer.Builder.relu;
import static com.codeberry.tadlib.provider.ProviderStore.shape;

public class TrainConfiguredConvMNISTMain {


    //    === Epoch 4946, 2021-02-19T03:18:42.269532300
    //    * Test acc: 0.9952229299363057
    //    (have had higher test acc in some earlier epochs)
    public static SequentialModel.Factory createModelFactory(ModelSize modelSize) {
        return cfgBuilder()
                .inputShape(shape(-1, 28, 28, 1))
                .layerBuilders(
                        batchNorm(),

                        blockDropout().blockSize(5).epochRange(0, 50).dropKeepRange(1.0, 0.95),
                        conv2d().biasParam(USE_BIAS).l2Lambda(0.01).filters(modelSize.filter0).kernelSize(3),
                        maxPool2d().size(2),
                        batchNorm(),
                        proportionalDropout().strength(0.3),
                        relu().leakyScale(0.01),

                        blockDropout().blockSize(3).epochRange(0, 100).dropKeepRange(1.0, 0.90),
                        conv2d().biasParam(NO_BIAS).l2Lambda(0.01).filters(modelSize.filter1).kernelSize(5),
                        maxPool2d().size(2),
                        batchNorm(),
                        proportionalDropout().strength(0.3),
                        relu().leakyScale(0.01),

                        blockDropout().blockSize(2).epochRange(0, 100).dropKeepRange(1.0, 0.85),
                        conv2d().biasParam(USE_BIAS).l2Lambda(0.01).filters(modelSize.filter2).kernelSize(3),
                        maxPool2d().size(2),
                        batchNorm(),
                        proportionalDropout().strength(0.3),
                        relu().leakyScale(0.01),

                        flatten(),
                        dense().biasParam(DenseLayer.BiasParam.NO_BIAS).l2Lambda(0.01).units(modelSize.hidden),
                        batchNorm(),
                        proportionalDropout().strength(0.3),
                        relu().leakyScale(0.01),

                        dropout().dropoutKeep(0.75),
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
