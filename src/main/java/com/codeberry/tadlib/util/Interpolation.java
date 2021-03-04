package com.codeberry.tadlib.util;

public class Interpolation {
    private final double valueStart;
    private final double valueEnd;
    private final int rangeStart;
    private final int rangeEnd;

    public Interpolation(double valueStart, double valueEnd, int rangeStart, int rangeEnd) {
        this.valueStart = valueStart;
        this.valueEnd = valueEnd;
        this.rangeStart = rangeStart;
        this.rangeEnd = rangeEnd;
    }

    public double interpolate(int position) {
        if (position < rangeStart) {
            return valueStart;
        }
        if (position >= rangeEnd) {
            return valueEnd;
        }
        int diff = position - rangeStart;

        int rangeLen = rangeEnd - rangeStart;
        double valLen = valueEnd - valueStart;

        return valueStart + valLen * diff / rangeLen;
    }
}
