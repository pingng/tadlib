package com.codeberry.tadlib.array;

import static java.lang.Math.abs;

public abstract class Comparison {

    public abstract boolean doubleIsTrue(double leftVal, double rightVal);

    public abstract Comparison getFlippedComparison();

    public abstract boolean intIsTrue(int l, int r);

    public abstract String getId();

    
    public static Comparison equalsWithDelta(double delta) {
        return new EqualsComparison(delta);
    }

    public static final Comparison greaterThan = new Comparison() {
        @Override
        public boolean doubleIsTrue(double leftVal, double rightVal) {
            return leftVal > rightVal;
        }

        @Override
        public Comparison getFlippedComparison() {
            return lessThan;
        }

        @Override
        public boolean intIsTrue(int l, int r) {
            return l > r;
        }

        @Override
        public String getId() {
            return ">";
        }
    };

    public static final Comparison lessThan = new Comparison() {
        @Override
        public boolean doubleIsTrue(double leftVal, double rightVal) {
            return leftVal < rightVal;
        }

        @Override
        public Comparison getFlippedComparison() {
            return greaterThan;
        }

        @Override
        public boolean intIsTrue(int l, int r) {
            return l < r;
        }

        @Override
        public String getId() {
            return "<";
        }

    };

    public static final Comparison lessThanOrEquals = new Comparison() {
        @Override
        public boolean doubleIsTrue(double leftVal, double rightVal) {
            return leftVal <= rightVal;
        }

        @Override
        public Comparison getFlippedComparison() {
            return greaterThanOrEquals;
        }

        @Override
        public boolean intIsTrue(int l, int r) {
            return l <= r;
        }

        @Override
        public String getId() {
            return "<=";
        }

    };
    public static final Comparison greaterThanOrEquals = new Comparison() {
        @Override
        public boolean doubleIsTrue(double leftVal, double rightVal) {
            return leftVal >= rightVal;
        }

        @Override
        public Comparison getFlippedComparison() {
            return lessThanOrEquals;
        }

        @Override
        public boolean intIsTrue(int l, int r) {
            return l >= r;
        }

        @Override
        public String getId() {
            return ">=";
        }

    };

    static class EqualsComparison extends Comparison {
        private final double delta;

        private EqualsComparison(double delta) {
            this.delta = delta;
        }

        @Override
        public Comparison getFlippedComparison() {
            return this;
        }

        @Override
        public boolean doubleIsTrue(double leftVal, double rightVal) {
            return abs(rightVal - leftVal) <= delta;
        }

        @Override
        public boolean intIsTrue(int l, int r) {
            return l == r;
        }

        @Override
        public String getId() {
            return "==";
        }

    }
}
