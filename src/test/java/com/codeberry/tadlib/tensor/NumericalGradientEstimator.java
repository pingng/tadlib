package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.nn.Model;
import com.codeberry.tadlib.provider.java.NDArray;
import com.codeberry.tadlib.provider.java.Shape;
import com.codeberry.tadlib.util.TrainingData;

import java.util.List;

class NumericalGradientEstimator {
    private final TrainingData trainingData;
    private final Model model;
    private final int rndSeed;

    public NumericalGradientEstimator(TrainingData trainingData, Model model, int rndSeed) {
        this.trainingData = trainingData;
        this.model = model;
        this.rndSeed = rndSeed;
    }

    public NDArray estimateParamGradIndex(int paramIndex) {
        ModelParamEstimator task = new ModelParamEstimator(rndSeed, model, paramIndex, trainingData);

        double[] doubles = task.estimate();
        List<Tensor> params = model.getParams();
        Tensor param = params.get(paramIndex);

        return new NDArray(doubles, new Shape(param.shape().toDimArray()));
    }

}
