package com.codeberry.tadlib.array;

import static java.lang.Math.abs;

public abstract class Comparison {
    public static Comparison equalsWithDelta(double delta) {
        return new EqualsComparison(delta);
    }

    public static Comparison greaterThan() {
        return new Comparison() {
            @Override
            public boolean doubleIsTrue(double leftVal, double rightVal) {
                return leftVal > rightVal;
            }

            @Override
            public Comparison getFlippedComparison() {
                return lessThan();
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
    }

    public static Comparison lessThan() {
        return new Comparison() {
            @Override
            public boolean doubleIsTrue(double leftVal, double rightVal) {
                return leftVal < rightVal;
            }

            @Override
            public Comparison getFlippedComparison() {
                return greaterThan();
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
    }

    public static Comparison greaterThanOrEquals() {
        return new Comparison() {
            @Override
            public boolean doubleIsTrue(double leftVal, double rightVal) {
                return leftVal >= rightVal;
            }

            @Override
            public Comparison getFlippedComparison() {
                return lessThanOrEquals();
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
    }

    public static Comparison lessThanOrEquals() {
        return new Comparison() {
            @Override
            public boolean doubleIsTrue(double leftVal, double rightVal) {
                return leftVal <= rightVal;
            }

            @Override
            public Comparison getFlippedComparison() {
                return greaterThanOrEquals();
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
    }

    public abstract boolean doubleIsTrue(double leftVal, double rightVal);

    public abstract Comparison getFlippedComparison();

    public abstract boolean intIsTrue(int l, int r);

    public abstract String getId();

    public double getDelta() {
        return 0.0;
    }

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
            double diff = rightVal - leftVal;
            return abs(diff) <= delta;
        }

        @Override
        public boolean intIsTrue(int l, int r) {
            return l == r;
        }

        @Override
        public String getId() {
            return "==";
        }

        @Override
        public double getDelta() {
            return delta;
        }
    }
}
