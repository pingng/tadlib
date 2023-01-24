package com.codeberry.tadlib.array.util;

import com.codeberry.tadlib.provider.java.JavaIntArray;
import com.codeberry.tadlib.provider.java.NDArray;

public abstract class SoftmaxUtils {
    public static NDArray calcSoftmaxCrossEntropyGradient(NDArray predicted, NDArray labelsOneHot) {
        JavaIntArray indices = labelsOneHot.argmax(-1);
        NDArray predAtTargetIndices = predicted.getAtIndicesOnAxis(indices, -1);
        NDArray change = predAtTargetIndices.add(-1);

        return predicted.withUpdateAtIndicesOnAxis(indices, -1, change);
    }
//    public static List<NDArray.ValueUpdate> getSoftmaxGradientUpdates(NDArray predicted, int[] indices, NDArray labelsOneHot, int dim) {
//        List<NDArray.ValueUpdate> updates = new ArrayList<>();
//        double[] predData = predicted.getInternalData();
//        double[] lblData = labelsOneHot.getInternalData();
//        addSoftmaxGradientUpdates(updates, predicted, predData, indices, labelsOneHot, lblData, dim);
//
//        return updates;
//    }
//
//    private static void addSoftmaxGradientUpdates(List<NDArray.ValueUpdate> updates, NDArray predicted, double[] predData, int[] indices, NDArray labelsOneHot, double[] lblData, int dim) {
//        int len = predicted.getShape().at(dim);
//        if (indices.length - dim == 1) {
//            // --- Find MAX index in last dim ---
//            int maxIndex = -1;
//            double max = Double.NEGATIVE_INFINITY;
//            for (int i = 0; i < len; i++) {
//                indices[dim] = i;
//                int lblOffset = labelsOneHot.getShape().calcDataIndex(indices);
//                double tgtVal = lblData[lblOffset];
//                if (tgtVal > max) {
//                    max = tgtVal;
//                    maxIndex = i;
//                }
//            }
//
//            // --- Only change value of MAX INDEX ---
//            indices[dim] = maxIndex;
//            int predOffset = predicted.getShape().calcDataIndex(indices);
//            double pred = predData[predOffset];
//            updates.add(fromIndices(pred - 1, predicted.getShape(), indices));
//        } else {
//            for (int i = 0; i < len; i++) {
//                indices[dim] = i;
//                addSoftmaxGradientUpdates(updates, predicted, predData, indices, labelsOneHot, lblData, dim + 1);
//            }
//        }
//    }
}
