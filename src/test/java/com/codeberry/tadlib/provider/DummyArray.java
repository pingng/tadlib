package com.codeberry.tadlib.provider;

import com.codeberry.tadlib.array.NDArray;
import com.codeberry.tadlib.array.NDIntArray;
import com.codeberry.tadlib.provider.java.JavaShape;

import java.util.List;
import java.util.Random;

class DummyArray implements NDArray {
    @Override
    public NDArray add(NDArray other) {
        return null;
    }

    @Override
    public NDArray add(double val) {
        return null;
    }

    @Override
    public NDArray mul(NDArray other) {
        return null;
    }

    @Override
    public NDArray div(NDArray other) {
        return null;
    }

    @Override
    public NDArray mul(double val) {
        return null;
    }

    @Override
    public NDArray div(double val) {
        return null;
    }

    @Override
    public NDArray conv2d(NDArray filter, int offsetY, int offsetX, int outHeight, int outWidth) {
        return null;
    }

    @Override
    public NDArray sum(Boolean[] dimsToCollapse, DimKeepRemove keepRemove) {
        return null;
    }

    @Override
    public NDArray conv2d(NDArray filter, int offsetY, int offsetX) {
        return null;
    }

    @Override
    public NDArray calcConv2dFilterGradient(NDArray input, NDArray filter) {
        return null;
    }

    @Override
    public NDArray matmul(NDArray b) {
        return null;
    }

    @Override
    public NDArray transpose(int... axes) {
        return null;
    }

    @Override
    public NDArray negate() {
        return null;
    }

    @Override
    public NDArray sqr() {
        return null;
    }

    @Override
    public NDArray sqrt() {
        return null;
    }

    @Override
    public NDArray rot180(int yAxis, int xAxis) {
        return null;
    }

    @Override
    public NDArray pow(double val) {
        return null;
    }

    @Override
    public MaxPool2dResult maxPool2d(int size) {
        return null;
    }

    @Override
    public NDArray maxPool2dGrad(MaxPool2dResult result) {
        return null;
    }

    @Override
    public ReluResult relu(double leakyScale) {
        return null;
    }

    @Override
    public NDArray softmax() {
        return null;
    }

    @Override
    public NDArray softMaxGrad(NDArray softmax, NDArray oneHotArray) {
        return null;
    }

    @Override
    public DropOutResult dropOut(Random rnd, double dropoutKeep) {
        return null;
    }

    @Override
    public NDArray withUpdates(List<ValueUpdate> updates) {
        return null;
    }

    @Override
    public NDArray clip(Double min, Double max) {
        return null;
    }

    @Override
    public NDArray log() {
        return null;
    }

    @Override
    public NDIntArray argmax(int axis) {
        return null;
    }

    @Override
    public NDArray getAtIndicesOnAxis(NDIntArray indices, int axis) {
        return null;
    }

    @Override
    public NDArray withUpdateAtIndicesOnAxis(NDIntArray indices, int axis, NDArray change) {
        return null;
    }

    @Override
    public JavaShape getShape() {
        return null;
    }

    @Override
    public NDArray reshape(int... dims) {
        return null;
    }

    @Override
    public Object toDoubles() {
        return null;
    }

    @Override
    public NDArray subArray(int fromBatchIndex, int fromOffset, int endBatchIndex, int toOffset) {
        return null;
    }

    @Override
    public NDArray normalOrderedCopy() {
        return null;
    }

    @Override
    public double[] getInternalData() {
        return new double[0];
    }

    @Override
    public double dataAt(int... indices) {
        return 0;
    }
}
