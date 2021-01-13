package com.codeberry.tadlib.util;

public class ClockTimer {
    private final long start;
    private final String name;

    public ClockTimer(String name) {
        this.name = name;
        this.start = System.currentTimeMillis();
    }

    public long getMillisPassed() {
        return System.currentTimeMillis() - start;
    }

    @Override
    public String toString() {
        long used = getMillisPassed();

        return "(Timer: " + name + ": " + used + "ms)";
    }

    public static ClockTimer timer() {
        return new ClockTimer("Timer");
    }

    public static ClockTimer timer(String name) {
        return new ClockTimer(name);
    }
}
