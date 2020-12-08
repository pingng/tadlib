package com.codeberry.tadlib.util;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;
import java.util.function.BiFunction;
import java.util.function.Function;

import static java.lang.Math.max;

public abstract class MultiThreadingSupport {
    private static volatile ForkJoinPool pool;

    public static void enableMultiThreading() {
        int processors = Runtime.getRuntime().availableProcessors();
        if (processors >= 2) {
            int threads = max(processors - 1, 2);
            pool = new ForkJoinPool(threads);
        }
    }

    public static void disableMultiThreading() {
        pool = null;
    }

    public static <R> R multiThreadingSupportRun(TaskRange totalRange, Function<TaskRange, R> task, BiFunction<R, R, R> merger) {
        if (pool == null) {
            return task.apply(totalRange);
        } else {
            return pool.invoke(new MyRecursiveTask<>(totalRange, task, merger));
        }
    }

    public static class TaskRange {
        private static final int DEFAULT_MIN_LENGTH = 1;

        public final int start;
        public final int end;
        public final int minLength;

        public TaskRange(int start, int end, int minLength) {
            this.start = start;
            this.end = end;
            this.minLength = minLength;
        }

        public static TaskRange taskRange(int start, int end) {
            return new TaskRange(start, end, DEFAULT_MIN_LENGTH);
        }

        public int size() {
            return end - start;
        }

        public int length() {
            return end - start;
        }

        public boolean isSmallEnoughForDirectWork() {
            return length() <= minLength;
        }

        public TaskRange withMinimumWorkLength(int minLen) {
            return new TaskRange(start, end, minLen);
        }

        public TaskRange leftPart() {
            return new TaskRange(start, start + length() / 2, minLength);
        }

        public TaskRange rightPart() {
            return new TaskRange(start + length() / 2, end, minLength);
        }

        @Override
        public String toString() {
            return "(" + start + ", " + end + ")";
        }
    }

    private static class MyRecursiveTask<R> extends RecursiveTask<R> {
        private final TaskRange range;
        private final Function<TaskRange, R> task;
        private final BiFunction<R, R, R> merger;

        public MyRecursiveTask(TaskRange range, Function<TaskRange, R> task, BiFunction<R, R, R> merger) {
            this.range = range;
            this.task = task;
            this.merger = merger;
        }

        @Override
        protected R compute() {
            if (range.isSmallEnoughForDirectWork()) {
                return task.apply(this.range);
            } else {
                TaskRange left = this.range.leftPart();
                TaskRange right = this.range.rightPart();

                R leftResult;
                R rightResult;
                MyRecursiveTask<R> leftTask = new MyRecursiveTask<>(left, task, merger);
                leftTask.fork();
                if (right.isSmallEnoughForDirectWork()) {
                    rightResult = task.apply(right);
                } else {
                    MyRecursiveTask<R> rightTask = new MyRecursiveTask<>(right, task, merger);
                    rightTask.fork();
                    rightResult = rightTask.join();
                }
                leftResult = leftTask.join();

                return merger.apply(leftResult, rightResult);
            }
        }
    }
}
