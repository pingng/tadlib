package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.array.TArray;
import com.codeberry.tadlib.example.TrainingData;
import com.codeberry.tadlib.nn.model.Model;

import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.stream.IntStream;

import static com.codeberry.tadlib.array.TArrayFactory.zeros;
import static java.lang.Math.min;
import static java.util.stream.Collectors.toList;

class NumericalGradientEstimator {
    private final TrainingData trainingData;
    private final Model model;
    private final int rndSeed;
    private final ExecutorService execSrv;
    private final int threadCount;

    public NumericalGradientEstimator(TrainingData trainingData, Model model, int rndSeed, ExecutorService execSrv, int threadCount) {
        this.trainingData = trainingData;
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
        List<Tensor> params = model.getParams();
        Tensor param = params.get(paramIndex);

        TArray retGrad = zeros(param.getShape());
        double[] tgt = retGrad.getInternalData();

        writeResultsInOrder(resultFutures, tgt);

        return retGrad;
    }

    private List<ArraySegmentEstimationTask> createArraySegmentEstimationTasks(int paramIndex) {
        List<Tensor> params = model.getParams();
        Tensor param = params.get(paramIndex);

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
                trainingData);
    }
}
