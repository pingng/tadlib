package com.codeberry.tadlib.mnist;

import com.codeberry.tadlib.nn.optimizer.RMSProp;
import com.codeberry.tadlib.nn.optimizer.SGD;
import com.codeberry.tadlib.nn.optimizer.schedule.FixedLearningRate;
import com.codeberry.tadlib.util.TrainingData;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import static com.codeberry.tadlib.mnist.FixedMNISTConvModel.Factory.Builder.factoryBuilder;
import static com.codeberry.tadlib.mnist.MNISTLoader.LoadParams.params;
import static com.codeberry.tadlib.mnist.SimpleTrainer.TrainParams.trainParams;
import static com.codeberry.tadlib.mnist.TrainConfiguredConvMNISTMain.createModelFactory;
import static com.codeberry.tadlib.nn.optimizer.schedule.DecayingLearningRate.decayingLearningRate;
import static com.codeberry.tadlib.nn.optimizer.schedule.SawToothSchedule.sawTooth;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class MnistTests {

    @Test
    public void HardCodedFullyConnected() {
        TrainingData trainingData = MNISTLoader.load(params()
                .loadRegularMNIST()
                .downloadWhenMissing(true)
                .trainingExamples(40_000 / 200)
                .testExamples(10_000 / 200));

        var m = new TrainHardCodedFullyConnectedMNISTModel(trainingData);
        m.trainForEpochs(
                //100
                8
        );
        assertTrue(m.accuracy > 0.2f);
    }

    @Disabled
    @Test
    void FixedConvTest() {
        var trainer = new SimpleTrainer(trainParams("FixedConv")
                .batchSize(32)
                .optimizer(new SGD(new FixedLearningRate(0.15)))
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

    @Test
    @Disabled
    void ConfiguredConvMNIST() {

        SimpleTrainer trainer = new SimpleTrainer(trainParams("Conf")
                .batchSize(64)
//                .batchSize(32)
//                .optimizer(new SGD(0.1))
//                .optimizer(new RMSProp(fixedLearningRate(0.0005)))
                .optimizer(new RMSProp(sawTooth(
                        decayingLearningRate(0.0005, 0.0000001, 10, 0.8),
                        0.3, 6)))
                .loaderParams(params()
                        .loadFashionMNIST()
                        .downloadWhenMissing(true)
                        .trainingExamples(40_000)
                        .testExamples(10_000))
                .modelFactory(createModelFactory(TrainConfiguredConvMNISTMain.ModelSize.HUGE_SIZE)));
//                .modelFactory(createModelFactory(ModelSize.NORMAL_SIZE)));
//                .modelFactory(createModelFactory(ModelSize.HUGE_SIZE)));

        trainer.trainEpochs(5000);

    }
}
