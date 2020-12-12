package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.array.TArray;
import com.codeberry.tadlib.example.TrainingData;
import com.codeberry.tadlib.example.mnist.MNISTFullyConnectedModel;
import com.codeberry.tadlib.nn.model.*;
import com.codeberry.tadlib.util.MatrixTestUtils;
import com.codeberry.tadlib.util.MultiThreadingSupport;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;
import java.util.concurrent.*;

import static com.codeberry.tadlib.example.mnist.MNISTFullyConnectedModel.Factory.Builder.factoryBuilder;
import static com.codeberry.tadlib.example.mnist.MNISTLoader.*;
import static com.codeberry.tadlib.example.mnist.TrainConfiguredConvMNISTMain.*;
import static java.lang.Math.*;
import static org.junit.jupiter.api.Assertions.fail;

public class LargeGradientTest {

    public static final int TEST_EPOCHS = 3;
    private int threadCount;
    private ExecutorService execSrv;
    public static final int TRAINING_EXAMPLES = 16;
    public static final double LEARNING_RATE = 0.1;

    @AfterEach
    public void destroy() {
        execSrv.shutdown();
    }

    private void initTestThreads(int threadCount) {
        this.threadCount = max(threadCount, 1);
        execSrv = Executors.newFixedThreadPool(this.threadCount);
    }

    @Test
    public void testGradientWhileTrainingModel_SingleThreaded() throws ExecutionException, InterruptedException {
        MultiThreadingSupport.disableMultiThreading();
        // use many threads since the calculations are single threaded
        initTestThreads(Runtime.getRuntime().availableProcessors() - 1);

        doTest();
    }

    @Test
    public void testGradientWhileTrainingModel_MultiThreaded() throws ExecutionException, InterruptedException {
        MultiThreadingSupport.enableMultiThreading();
        // use a single test thread since the calculations are multi threaded
        initTestThreads(1);

        doTest();
    }

    private void doTest() throws ExecutionException, InterruptedException {
        Random rand = new Random(4);
        TrainingData data = generate(rand, TRAINING_EXAMPLES);

        ModelFactory modelFactory =
                createConvolutionModelFactory()
                //createFullyConnectedModelFactory()
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

    private double testGradients(TrainingData trainingData, Model model, int epoch) throws ExecutionException, InterruptedException {
        model.calcGradient(new Random(epoch), trainingData);
        List<TArray> gradients = model.getGradients();

        ErrorValidator errorValidator = new ErrorValidator();
        NumericalGradientEstimator estimator = new NumericalGradientEstimator(trainingData, model, epoch, execSrv, threadCount);
        for (int i = 0; i < gradients.size(); i++) {
            TArray backpropGrad = gradients.get(i);
            if (backpropGrad != null) {
                TArray numericalEstimatedGrad = estimator.estimateParamGradIndex(i);

                errorValidator.validateAndAccumulateErr(backpropGrad, numericalEstimatedGrad);
            } else {
                System.err.println("WARN: missing gradient for param index: " + i);
            }
        }

        return errorValidator.getAvgError();
    }

    private static class ErrorValidator {
        private static final double IDEAL_ERR_ASPECT = 5e-7;
        private static final double MAX_ERR_ASPECT = 0.055;
        private static final int MAX_CONSECUTIVE_LARGE_ASPECT_TIMES = 4;

        private int largeAspectCount = 0;
        private int sumErrorCount;
        private double sumError;

        public boolean isNotOk(double errAspect) {
            return errAspect >= IDEAL_ERR_ASPECT;
        }

        public void failWhenTooLargeOrTooOften(double errAspect) {
            if (isNotOk(errAspect)) {
                largeAspectCount++;
            } else {
                largeAspectCount = 0;
            }
            System.out.println("errAspect = " + errAspect + " largeAspectCount=" + largeAspectCount);

            if (largeAspectCount > MAX_CONSECUTIVE_LARGE_ASPECT_TIMES) {
                fail("Err Aspect was big too often!");
            }
            if (errAspect > MAX_ERR_ASPECT) {
                fail("Too large error aspect!");
            }
        }

        private void validateAndAccumulateErr(TArray backpropGrad, TArray numericalGrad) {
            double errAspect = MatrixTestUtils.calcErrAspect(backpropGrad.toDoubles(), numericalGrad.toDoubles());

            if (isNotOk(errAspect)) {
                System.out.println("Backprop:  " + backpropGrad);
                System.out.println("Numerical: " + numericalGrad);
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
