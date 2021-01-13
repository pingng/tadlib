package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.array.NDArray;
import com.codeberry.tadlib.example.TrainingData;
import com.codeberry.tadlib.example.mnist.MNISTFullyConnectedModel;
import com.codeberry.tadlib.nn.model.*;
import com.codeberry.tadlib.nn.model.optimizer.SGD;
import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.java.JavaProvider;
import com.codeberry.tadlib.provider.opencl.OpenCLProvider;
import com.codeberry.tadlib.util.MatrixTestUtils;
import com.codeberry.tadlib.util.StringUtils;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import static com.codeberry.tadlib.example.mnist.MNISTFullyConnectedModel.Factory.Builder.factoryBuilder;
import static com.codeberry.tadlib.example.mnist.MNISTLoader.*;
import static com.codeberry.tadlib.example.mnist.TrainConfiguredConvMNISTMain.*;
import static com.codeberry.tadlib.provider.java.JavaProvider.ThreadMode.MULTI_THREADED;
import static com.codeberry.tadlib.provider.java.JavaProvider.ThreadMode.SINGLE_THREADED;
import static org.junit.jupiter.api.Assertions.fail;

public class LargeGradientTest {

    public static final int TEST_EPOCHS = 3;
    public static final int TRAINING_EXAMPLES = 16;
    public static final double LEARNING_RATE = 0.1;

    @Test
    public void testGradientWhileTrainingModel_SingleThreaded() {
        ProviderStore.setProvider(new JavaProvider(SINGLE_THREADED));

        doTest();
    }

    @Test
    public void testGradientWhileTrainingModel_MultiThreaded() {
        ProviderStore.setProvider(new JavaProvider(MULTI_THREADED));

        doTest();
    }

    @Test
    public void testGradientWhileTrainingModel_OpenCL() {
        ProviderStore.setProvider(new OpenCLProvider());

        doTest();
    }

    private void doTest() {
        Random rand = new Random(4);
        TrainingData data = generate(rand, TRAINING_EXAMPLES);

        ModelFactory modelFactory =
                createConvolutionModelFactory()
//                createFullyConnectedModelFactory()
                ;

        Model model = modelFactory.createModel();
        System.out.println("Model: " + model.getClass().getSimpleName());

        double sumError = 0;
        for (int epoch = 0; epoch <= TEST_EPOCHS; epoch++) {
            System.out.println("=== Epoch " + epoch);

            sumError += testGradients(data, model, epoch);
            System.out.println("* avgError=" + (sumError / (epoch + 1)));

            trainModel(data, model, epoch);
        }
    }

    private static MNISTFullyConnectedModel.Factory createFullyConnectedModelFactory() {
        return factoryBuilder()
                .hiddenNeurons(16)
                .weightInitRandomSeed(4)
                .build();
    }

    private static SequentialModel.Factory createConvolutionModelFactory() {
        return createModelFactory(ModelSize.TINY);
    }

    private double testGradients(TrainingData trainingData, Model model, int epoch) {
        model.calcGradient(new Random(epoch), trainingData);
        List<NDArray> gradients = model.getGradients();
        List<Object> gradientDoubles = gradients.stream()
                .map(NDArray::toDoubles)
                .collect(Collectors.toList());

        ErrorValidator errorValidator = new ErrorValidator();
        NumericalGradientEstimator estimator = new NumericalGradientEstimator(trainingData, model, epoch);
        for (int i = 0; i < gradientDoubles.size(); i++) {
            Object autoGradGradient = gradientDoubles.get(i);
            NDArray numericalEstimatedGrad = estimator.estimateParamGradIndex(i);

            errorValidator.validateAndAccumulateErr(autoGradGradient, numericalEstimatedGrad);
        }

        return errorValidator.getAvgError();
    }

    private static class ErrorValidator {
        private static final double IDEAL_ERR_ASPECT = 5e-7;
        private static final double MAX_ERR_ASPECT = 0.055;
        private static final int MAX_CONSECUTIVE_LARGE_ASPECT_TIMES = 4;

        private int consecutiveLargeAspectCount = 0;
        private int sumErrorCount;
        private double sumError;

        public boolean isNotOk(double errAspect) {
            return errAspect >= IDEAL_ERR_ASPECT;
        }

        public void failWhenTooLargeOrTooOften(double errAspect) {
            if (isNotOk(errAspect)) {
                consecutiveLargeAspectCount++;
            } else {
                consecutiveLargeAspectCount = 0;
            }
            System.out.println("errAspect = " + errAspect + " consecutiveLargeAspectCount=" + consecutiveLargeAspectCount);

            if (consecutiveLargeAspectCount > MAX_CONSECUTIVE_LARGE_ASPECT_TIMES) {
                fail("Err Aspect was big too often!");
            }
            if (errAspect > MAX_ERR_ASPECT) {
                fail("Too large error aspect!");
            }
        }

        private void validateAndAccumulateErr(Object autoGradGradient, NDArray numericalGrad) {
            Object numericalGradient = numericalGrad.toDoubles();
            double errAspect = MatrixTestUtils.calcErrAspect(
                    autoGradGradient,
                    numericalGradient);

            if (isNotOk(errAspect)) {
                System.err.println("ErrAspect: " + errAspect + " (ideally below " + IDEAL_ERR_ASPECT + ")");
                System.err.println("Backprop:  " + StringUtils.toString(numericalGrad.getShape(), autoGradGradient));
                System.err.println("Numerical: " + StringUtils.toString(numericalGrad.getShape(), numericalGradient));
            }

            failWhenTooLargeOrTooOften(errAspect);

            this.sumError += errAspect;
            this.sumErrorCount++;
        }

        public double getAvgError() {
            return sumError / sumErrorCount;
        }
    }

    private void trainModel(TrainingData trainingData, Model model, int epoch) {
        Random dropRnd = new Random(epoch);
        TrainStats stats = new TrainStats();

        Model.PredictionAndLosses pl = model.trainSingleIteration(dropRnd, trainingData, new SGD(0.1));

        stats.accumulate(pl, trainingData.yTrain);
        System.out.println("Trained: " + stats);
    }
}
