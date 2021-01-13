package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.provider.java.JavaArray;
import com.codeberry.tadlib.provider.java.JavaShape;
import com.codeberry.tadlib.example.TrainingData;
import com.codeberry.tadlib.nn.model.Model;

import java.util.List;

import static java.lang.Math.min;

class NumericalGradientEstimator {
    private final TrainingData trainingData;
    private final Model model;
    private final int rndSeed;

    public NumericalGradientEstimator(TrainingData trainingData, Model model, int rndSeed) {
        this.trainingData = trainingData;
        this.model = model;
        this.rndSeed = rndSeed;
    }

    public JavaArray estimateParamGradIndex(int paramIndex) {
        ModelParamEstimator task = new ModelParamEstimator(rndSeed, model, paramIndex, trainingData);

        double[] doubles = task.estimate();
        List<Tensor> params = model.getParams();
        Tensor param = params.get(paramIndex);

        return new JavaArray(doubles, new JavaShape(param.getShape().toDimArray()));
    }

}
