package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.array.TArray;
import com.codeberry.tadlib.example.mnist.MNISTConvModel;

import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.stream.IntStream;

import static java.lang.Math.min;
import static java.util.stream.Collectors.*;

class NumericalGradientEstimator {
    private final Tensor xTrain;
    private final Tensor yTrain;
    private final MNISTConvModel model;
    private final int rndSeed;
    private final ExecutorService execSrv;
    private final int threadCount;

    public NumericalGradientEstimator(Tensor xTrain, Tensor yTrain, MNISTConvModel model, int rndSeed, ExecutorService execSrv, int threadCount) {
        this.xTrain = xTrain;
        this.yTrain = yTrain;
        this.model = model;
        this.rndSeed = rndSeed;
        this.execSrv = execSrv;
        this.threadCount = threadCount;
    }

    public TArray estimateParamGradIndex(int paramIndex) throws ExecutionException, InterruptedException {

        List<ArraySegmentEstimationTask> tasks = createArraySegmentEstimationTasks(paramIndex);

        System.out.println("Est tasks: " + tasks);
        List<Future<double[]>> resultFutures = execSrv.invokeAll(tasks);

        return mergeResultsToSingleGradientResult(paramIndex, resultFutures);
    }

    private TArray mergeResultsToSingleGradientResult(int paramIndex, List<Future<double[]>> resultFutures) throws InterruptedException, ExecutionException {
        Tensor param = model.getParam(paramIndex);

        TArray retGrad = TArray.zeros(param.getShape());
        double[] tgt = retGrad.getInternalData();

        writeResultsInOrder(resultFutures, tgt);

        return retGrad;
    }

    private List<ArraySegmentEstimationTask> createArraySegmentEstimationTasks(int paramIndex) {
        Tensor param = model.getParam(paramIndex);

        int paramLen = param.getShape().size;
        int threadsToUse = min(threadCount, paramLen);

        return IntStream.range(0, threadsToUse)
                .mapToObj(i -> new int[]{
                        i * paramLen / threadsToUse, // fromIndex
                        min((i + 1) * paramLen / threadsToUse, paramLen) // toIndex
                })
                .map(indices -> newArraySegmentEstimationTask(paramIndex, indices[0], indices[1]))
                .collect(toList());
    }

    private static void writeResultsInOrder(List<Future<double[]>> resultFutures, double[] tgt) throws InterruptedException, ExecutionException {
        int offset = 0;
        for (Future<double[]> rF : resultFutures) {
            double[] grad = rF.get();
            System.arraycopy(grad, 0, tgt, offset, grad.length);

            offset += grad.length;
        }
    }

    private ArraySegmentEstimationTask newArraySegmentEstimationTask(int paramIndex, int fromIndex, int endIndex) {
        return new ArraySegmentEstimationTask(rndSeed, model.copy(), paramIndex, fromIndex, endIndex,
                xTrain, yTrain);
    }
}
