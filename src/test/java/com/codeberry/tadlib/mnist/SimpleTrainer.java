package com.codeberry.tadlib.mnist;

import com.codeberry.tadlib.nn.Model;
import com.codeberry.tadlib.nn.Model.IterationInfo;
import com.codeberry.tadlib.nn.ModelFactory;
import com.codeberry.tadlib.nn.optimizer.Optimizer;
import com.codeberry.tadlib.provider.java.Shape;
import com.codeberry.tadlib.tensor.Tensor;
import com.codeberry.tadlib.util.Batch;
import com.codeberry.tadlib.util.TrainStats;
import com.codeberry.tadlib.util.TrainingData;
import com.codeberry.tadlib.util.memory.LeakDetector;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static com.codeberry.tadlib.util.AccuracyUtils.softmaxAccuracy;
import static com.codeberry.tadlib.util.memory.DisposalRegister.Disposable;
import static com.codeberry.tadlib.util.memory.DisposalRegister.modelIteration;
import static java.util.Collections.emptyList;

public class SimpleTrainer {

    private final TrainParams params;
    private final TrainLogger logger = new TrainLogger();
    public final TrainingData data;
    private final Model model;
    private final Optimizer optimizer;
    private volatile RunPerformance performance = RunPerformance.FULL_SPEED;

    public SimpleTrainer(TrainParams params) {
        this.params = params;

        //System.out.println(StringUtils.toJson(params));

        data = MNISTLoader.load(params.loaderParams);
        optimizer = params.optimizer;
        model = params.modelFactory.createModel();

        //addSystemTray();
    }

    public void trainEpochs(int epochs) {
        int numberOfBatches = data.calcTrainingBatchCountOfSize(params.batchSize);
        int numberOfTestBatches = data.calcTestBatchCountOfSize(params.batchSize);
        System.out.println("trainBatches=" + numberOfBatches + " testBatches=" + numberOfTestBatches);
        printParamCount();

        LeakDetector.reset();

        Model.TrainInfo trainInfo = null;
        Random rnd = new Random(4);
        for (int epoch = 0; epoch <= epochs; epoch++) {
            System.out.println("=== Epoch " + epoch + ", " + LocalDateTime.now());
            TrainStats stats = new TrainStats();
            for (int batchId = 0; batchId < numberOfBatches; batchId++) {
                handlePause();

                IterationInfo iterationInfo = new IterationInfo(epoch, batchId, numberOfBatches, trainInfo);
                optimizer.learningRateSchedule().beforeBatch(iterationInfo);

                modelIteration(() -> trainBatch(rnd, stats, iterationInfo));
            }
            System.out.println(stats);

            double[] testAccuracy = new double[1];
            for (int batchId = 0; batchId < numberOfTestBatches; batchId++) {

                IterationInfo iterationInfo = new IterationInfo(-1, batchId, numberOfTestBatches);
                modelIteration(() -> {
                    testAccuracy[0] += testBatch(iterationInfo);

                    return emptyList();
                });
            }
            double testAcc = testAccuracy[0] / numberOfTestBatches;
            System.out.println("* Test acc: " + testAcc);

            trainInfo = new Model.TrainInfo(stats.asOutputStats(), new Model.OutputStats(-1, testAcc));
            LeakDetector.printOldObjectsAndIncreaseObjectAge();
        }
    }

    private void printParamCount() {
        long paramCount = 0;

        List<Tensor> params = model.getParams();
        for (Tensor param : params) {
            Shape javaShape = param.shape();
            paramCount += javaShape.size;
        }

        System.out.println("Total (main) params: " + paramCount);
    }

    private double testBatch(IterationInfo iterationInfo) {
        Batch testBatch = data.getTestBatch(iterationInfo.batchIndex, params.batchSize);

        Tensor predict = model.predict(testBatch.input, iterationInfo);

        return softmaxAccuracy(testBatch.output, predict);
    }

    private List<Disposable> trainBatch(Random rnd, TrainStats stats, IterationInfo iterationInfo) {
        Batch batchData = data.getTrainingBatch(iterationInfo.batchIndex, params.batchSize);

        Model.PredictionAndLosses pl = model.train(rnd, batchData, optimizer, iterationInfo);

        stats.accumulate(pl, batchData.output);

        logger.log(iterationInfo.batchIndex, iterationInfo.batchCount, model, stats);

        return getKeepInMemoryDisposables();
    }

    /**
     * @return disposables that must be kept (not disposed/released) after a training/predict iteration.
     */
    private List<Disposable> getKeepInMemoryDisposables() {
        List<Disposable> objects = new ArrayList<>(model.getKeepInMemoryDisposables());
        for (Tensor param : model.getParams()) {
            objects.addAll(param.getDisposables());
        }
        objects.addAll(optimizer.getKeepInMemoryDisposables());
        return objects;
    }

    private void handlePause() {
        while (performance == RunPerformance.PAUSED) {
            try {
                Thread.sleep(1000);
            } catch (InterruptedException ignore) {
            }
        }
        if (performance == RunPerformance.SLOW) {
            try {
                Thread.sleep(100);
            } catch (InterruptedException ignore) {
            }
        }
    }

//    private void addSystemTray() {
//        //Check the SystemTray is supported
//        if (!SystemTray.isSupported()) {
//            System.out.println("SystemTray is not supported");
//            return;
//        }
//        final PopupMenu popup = new PopupMenu();
//        BufferedImage image = new BufferedImage(16, 16, BufferedImage.TYPE_INT_RGB);
//        Graphics g = image.getGraphics();
//        g.fill3DRect(0, 0, 16, 16, true);
//        g.setColor(Color.BLACK);
//        g.drawString("T", 4, 12);
//
//        final TrayIcon trayIcon =
//                new TrayIcon(image, "Trainer");
//        final SystemTray tray = SystemTray.getSystemTray();
//
//        // Create a pop-up menu components
//        MenuItem gcItem = new MenuItem("Garbage Collect");
//        gcItem.addActionListener(ev -> {
//            System.out.println("GC!!!");
//            System.gc();
//        });
//        MenuItem pause = new MenuItem("Pause");
//        pause.addActionListener(ev -> {
//            this.performance = RunPerformance.PAUSED;
//            System.out.println(this.performance + " (" + LocalDateTime.now() + ")");
//        });
//        MenuItem slow = new MenuItem("Slow");
//        slow.addActionListener(ev -> {
//            this.performance = RunPerformance.SLOW;
//            System.out.println(this.performance + " (" + LocalDateTime.now() + ")");
//        });
//        MenuItem fullSpeed = new MenuItem("Full speed");
//        fullSpeed.addActionListener(ev -> {
//            this.performance = RunPerformance.FULL_SPEED;
//            System.out.println(this.performance + " (" + LocalDateTime.now() + ")");
//        });
//
//        popup.add(params.name + ": " + ProviderStore.getProviderShortDescription());
//        popup.addSeparator();
//        popup.add(fullSpeed);
//        popup.add(slow);
//        popup.add(pause);
//        popup.addSeparator();
//        popup.add(gcItem);
//
//        trayIcon.setPopupMenu(popup);
//
//        try {
//            tray.add(trayIcon);
//        } catch (AWTException e) {
//            System.out.println("TrayIcon could not be added.");
//        }
//    }

    static class TrainLogger {
        static final int OUTPUT_BATCHES = 200;
        int batchProgress = 0;
        long lastMillis;

        public void log(int batchId, int numberOfBatches, Model model, TrainStats stats) {
            if (lastMillis == 0) {
                lastMillis = System.currentTimeMillis();
            }
            int batchIdMod = batchId % OUTPUT_BATCHES;
            if (batchIdMod == 0) {
                batchProgress = 0;
                System.out.println("- Batch " + batchId + "/" + numberOfBatches);
                System.out.println("  " + stats);
                System.out.println(model.getTrainingLogText());
            } else {
                int progress10Percent = batchIdMod * 10 / OUTPUT_BATCHES;
                if (progress10Percent != batchProgress) {
                    long now = System.currentTimeMillis();
                    long used = lastMillis > 0 ? now - lastMillis : -1;
                    System.out.println(progress10Percent * 10 + "%" + (used != -1 ? " used: " + (used / 1000.0) : ""));
                    batchProgress = progress10Percent;
                    lastMillis = now;
                }
            }
        }
    }

    public static class TrainParams {
        private final String name;
        MNISTLoader.LoadParams loaderParams;
        ModelFactory modelFactory;
        Optimizer optimizer;
        int batchSize;

        private TrainParams(String name) {
            this.name = name;
        }

        public static TrainParams trainParams(String name) {
            return new TrainParams(name);
        }

        TrainParams loaderParams(MNISTLoader.LoadParams loaderParams) {
            this.loaderParams = loaderParams;
            return this;
        }

        TrainParams batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        TrainParams optimizer(Optimizer optimizer) {
            this.optimizer = optimizer;
            return this;
        }

        TrainParams modelFactory(ModelFactory modelFactory) {
            this.modelFactory = modelFactory;
            return this;
        }
    }

    private enum RunPerformance {
        FULL_SPEED, SLOW, PAUSED
    }
}
