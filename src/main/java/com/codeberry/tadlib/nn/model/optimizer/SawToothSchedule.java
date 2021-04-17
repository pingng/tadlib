package com.codeberry.tadlib.nn.model.optimizer;

import com.codeberry.tadlib.nn.model.Model;
import com.codeberry.tadlib.util.Interpolation;

public class SawToothSchedule implements LearningRateSchedule {

    private final LearningRateSchedule schedule;
    private final int periodEpochs;
    private final Interpolation leftPart;
    private final Interpolation rightPart;

    private int epoch = Integer.MIN_VALUE;

    public SawToothSchedule(LearningRateSchedule schedule, double change, int periodEpochs) {
        this.schedule = schedule;
        this.periodEpochs = periodEpochs;

        this.leftPart = new Interpolation(1. + change, 1. - change, 0, periodEpochs / 2);
        this.rightPart = new Interpolation(1. - change, 1. + change, periodEpochs / 2, periodEpochs);
    }

    public static SawToothSchedule sawTooth(LearningRateSchedule schedule, double change, int periodEpochs) {
        return new SawToothSchedule(schedule, change, periodEpochs);
    }

    @Override
    public void beforeBatch(Model.IterationInfo iterationInfo) {
        schedule.beforeBatch(iterationInfo);

        if (this.epoch != iterationInfo.epoch) {
            this.epoch = iterationInfo.epoch;
            System.out.println("Lr scale: " + calcScale() + ", lr=" + getLearningRate());
        }
    }

    @Override
    public double getLearningRate() {
        return schedule.getLearningRate() * calcScale();
    }

    private double calcScale() {
        double mod;

        int e = epoch % periodEpochs;
        if (e < periodEpochs / 2) {
            mod = leftPart.interpolate(e);
        } else {
            mod = rightPart.interpolate(e);
        }
        return mod;
    }
}
