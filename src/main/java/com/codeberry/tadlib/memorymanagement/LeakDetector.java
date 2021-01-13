package com.codeberry.tadlib.memorymanagement;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.LinkedHashSet;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicLong;

import static java.lang.Integer.toHexString;

/**
 * Logs object allocations and releases. Objects are logged by a long id. They er randomly selected to be logged.
 * This keeps the overhead to a minimum.
 *
 * Each id has allocation stacktraces attached to it, including a creation/birth 'year'. Caller can increment the
 * 'object year' and print old ids that has not been fully released.
 */
public class LeakDetector {
    private static int maxAge = 3;
    private static int maxEntries = 256;
    private static final ConcurrentMap<Long, AllocationInfo> infoPerObjectId = new ConcurrentHashMap<>();
    private static final ConcurrentMap<String, String> ignoreKeys = new ConcurrentHashMap<>();
    private static final AtomicLong objectYear = new AtomicLong();

    public static void allocate(long objectId) {
        if (infoPerObjectId.size() < maxEntries) {
            infoPerObjectId.compute(objectId, (id, info) -> {
                if (info != null) {
                    info.addAllocation(new StackTraceInfo(new Exception()));
                    return info;
                } else {
                    if (Math.random() < 0.01) {
                        StackTraceInfo traceInfo = new StackTraceInfo(new Exception());
                        if (ignoreKeys.containsKey(traceInfo.key)) {
                            AllocationInfo _info = new AllocationInfo(id, objectYear);
                            _info.addAllocation(traceInfo);
                            return _info;
                        }
                    }
                    return null;
                }
            });
        }
    }

    public static void release(long objectId) {
        AllocationInfo info = infoPerObjectId.get(objectId);
        if (info != null) {
            info.addRelease(new StackTraceInfo(new Exception()),
                    () -> infoPerObjectId.remove(objectId));
        }
    }

    public static void printOldObjectsAndIncreaseObjectAge() {
        for (AllocationInfo info : infoPerObjectId.values()) {
            if (info.getAge() >= maxAge) {
                info.enableReleasePrint();
                System.err.println(info);
            }
        }
        objectYear.incrementAndGet();
    }

    public static void reset() {
        infoPerObjectId.clear();
    }

    public static void ignore(String desc, String... keys) {
        for (String key : keys) {
            ignoreKeys.put(key, desc);
        }
    }

    private static class StackTraceInfo {
        private final String key;
        private final String stackTrace;

        private StackTraceInfo(Exception e) {
            this.stackTrace = toString(e);
            this.key = toHexString(this.stackTrace.hashCode()) + "|" + toHexString(this.stackTrace.length());
        }

        private static String toString(Exception e) {
            StringWriter writer = new StringWriter();
            e.printStackTrace(new PrintWriter(writer));
            return writer.toString();
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            StackTraceInfo that = (StackTraceInfo) o;
            return stackTrace.equals(that.stackTrace);
        }

        @Override
        public int hashCode() {
            return stackTrace.hashCode();
        }

        @Override
        public String toString() {
            return key + ":" + stackTrace;
        }
    }

    private static class AllocationInfo {
        private final long objectId;

        private final Set<StackTraceInfo> allocStackTraces = new LinkedHashSet<>();
        private final Set<StackTraceInfo> releaseStackTraces = new LinkedHashSet<>();
        private final AtomicLong objectYear;
        private final long birthYear;
        private int allocationCount;
        private int releaseCount;
        private boolean printOnRelease;

        public AllocationInfo(long objectId, AtomicLong objectYear) {
            this.objectId = objectId;
            this.objectYear = objectYear;
            this.birthYear = this.objectYear.get();
        }

        @Override
        public String toString() {
            return "--- Unreleased Object " + objectId + " (" +
                    getAge() + " yearsOld)\n" +
                    "#Alloc/#release: " + allocationCount + "/" + releaseCount +
                    "\nAlloc:\n" + allocStackTraces +
                    "\nRelease:\n" + releaseStackTraces;
        }

        private long getAge() {
            return objectYear.get() - birthYear;
        }

        public synchronized void addAllocation(StackTraceInfo traceInfo) {
            allocStackTraces.add(traceInfo);
            allocationCount++;
        }

        public synchronized void addRelease(StackTraceInfo traceInfo, Runnable onFullyReleased) {
            releaseStackTraces.add(traceInfo);
            releaseCount++;

            if (releaseCount == allocationCount) {
                onFullyReleased.run();

                if (printOnRelease) {
                    System.err.println("Eventually released " + objectId + "\n" +
                            traceInfo);
                }
            }
        }

        public synchronized void enableReleasePrint() {
            this.printOnRelease = true;
        }
    }
}
