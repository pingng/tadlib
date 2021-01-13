package com.codeberry.tadlib.example.mnist;

import com.codeberry.tadlib.memorymanagement.LeakDetector;
import com.codeberry.tadlib.example.TrainingData;
import com.codeberry.tadlib.nn.model.Model;
import com.codeberry.tadlib.nn.model.ModelFactory;
import com.codeberry.tadlib.nn.model.optimizer.Optimizer;
import com.codeberry.tadlib.nn.model.TrainStats;
import com.codeberry.tadlib.tensor.Tensor;
import com.codeberry.tadlib.util.StringUtils;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static com.codeberry.tadlib.memorymanagement.DisposalRegister.*;
import static com.codeberry.tadlib.util.AccuracyUtils.softmaxAccuracy;
import static java.util.Collections.emptyList;

public class SimpleTrainer {

    private final TrainParams params;
    private final TrainLogger logger = new TrainLogger();
    private final TrainingData trainingData;
    private final Model model;
    private final Optimizer optimizer;
    private volatile boolean paused;

    public SimpleTrainer(TrainParams params) {
        this.params = params;

        System.out.println(StringUtils.toJson(params));

        trainingData = MNISTLoader.load(params.loaderParams);
        optimizer = params.optimizer;
        model = params.modelFactory.createModel();

        addSystemTray();
    }

    public void trainEpochs(int epochs) {
        int numberOfBatches = trainingData.calcTrainingBatchCountOfSize(params.batchSize);
        System.out.println("numberOfBatches = " + numberOfBatches);

        LeakDetector.reset();

        Random rnd = new Random(4);
        for (int epoch = 0; epoch <= epochs; epoch++) {
            System.out.println("=== Epoch " + epoch);
            TrainStats stats = new TrainStats();
            for (int batchId = 0; batchId < numberOfBatches; batchId++) {
                handlePause();

                int finalBatchId = batchId;
                modelIteration(() -> trainBatch(finalBatchId, numberOfBatches, rnd, stats));
            }
            System.out.println(stats);

            modelIteration(() -> {
                Tensor predict = model.predict(trainingData.xTest);
                double testAccuracy = softmaxAccuracy(trainingData.yTest, predict);
                System.out.println("* Test acc: " + testAccuracy);

                return emptyList();
            });
            LeakDetector.printOldObjectsAndIncreaseObjectAge();
        }
    }

    private List<DisposableContainer<? extends Disposable>> trainBatch(int finalBatchId, int numberOfBatches, Random rnd, TrainStats stats) {
        TrainingData batchData = trainingData.getTrainingBatch(finalBatchId, params.batchSize);

        Model.PredictionAndLosses pl = model.trainSingleIteration(rnd, batchData, optimizer);

        stats.accumulate(pl, batchData.yTrain);

        logger.log(finalBatchId, numberOfBatches, model, stats);

        return getContainersOfResourcesToKeep();
    }

    /**
     * @return containers of disposable that must be kept (not disposed/released) after a training/predict iteration.
     */
    private List<DisposableContainer<? extends Disposable>> getContainersOfResourcesToKeep() {
        List<DisposableContainer<? extends Disposable>> objects = new ArrayList<>();
        objects.add(model::getNonDisposedObjects);
        objects.addAll(model.getParams());
        objects.addAll(optimizer.getNonDisposedContainers());
        return objects;
    }

    private void handlePause() {
        while (this.paused) {
            try {
                Thread.sleep(1000);
            } catch (InterruptedException ignore) {
            }
        }
    }

    private void addSystemTray() {
        //Check the SystemTray is supported
        if (!SystemTray.isSupported()) {
            System.out.println("SystemTray is not supported");
            return;
        }
        final PopupMenu popup = new PopupMenu();
        BufferedImage image = new BufferedImage(16, 16, BufferedImage.TYPE_INT_RGB);
        Graphics g = image.getGraphics();
        g.fill3DRect(0, 0, 16, 16, true);
        g.setColor(Color.BLACK);
        g.drawString("T", 4, 12);

        final TrayIcon trayIcon =
                new TrayIcon(image, "Trainer");
        final SystemTray tray = SystemTray.getSystemTray();

        // Create a pop-up menu components
        MenuItem gcItem = new MenuItem("Garbage Collect");
        gcItem.addActionListener(ev -> {
            System.out.println("GC!!!");
            System.gc();
        });
        MenuItem togglePause = new MenuItem("Pause/Unpause");
        togglePause.addActionListener(ev -> {
            this.paused = !this.paused;
            System.out.println("Paused: " + this.paused + " (" +
                    LocalDateTime.now() + ")");
        });

        popup.add(togglePause);
        popup.addSeparator();
        popup.add(gcItem);

        trayIcon.setPopupMenu(popup);

        try {
            tray.add(trayIcon);
        } catch (AWTException e) {
            System.out.println("TrayIcon could not be added.");
        }
    }

    static class TrainLogger {
        static  final int OUTPUT_BATCHES = 200;
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
                    System.out.println(progress10Percent * 10 + "%" + (used != -1 ? " used: " + ((double) used / 1000.0) : ""));
                    batchProgress = progress10Percent;
                    lastMillis = now;
                }
            }
        }
    }

    static class TrainParams {
        MNISTLoader.LoadParams loaderParams;
        ModelFactory modelFactory;
        Optimizer optimizer;
        int batchSize;

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
}
