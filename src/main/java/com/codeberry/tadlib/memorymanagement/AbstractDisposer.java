package com.codeberry.tadlib.memorymanagement;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.concurrent.Callable;

import static java.lang.Long.toHexString;

public abstract class AbstractDisposer {
    private static final boolean TRACK_GC_RELEASE = false;

    // The same lock object is used by all views
    protected final Object memObjLock;
    protected boolean released;

    private final String initStackTrace;
    private volatile Runnable callback;

    {
        if (TRACK_GC_RELEASE) {
            initStackTrace = toString(new Exception());
        } else {
            initStackTrace = null;
        }
    }

    private static String toString(Exception e) {
        StringWriter writer = new StringWriter();
        e.printStackTrace(new PrintWriter(writer));
        return writer.toString();
    }

    protected AbstractDisposer() {
        this.memObjLock = new Object();
    }

    /**
     * Create disposer to the same resource using the same lock.
     */
    protected AbstractDisposer(AbstractDisposer parent) {
        this.memObjLock = parent.memObjLock;
    }

    void releaseByGc() {
        release(false);
    }

    public void release() {
        release(true);
    }

    private void release(boolean manualCall) {
        synchronized (memObjLock) {
            if (!released) {
                Runnable cb = this.callback;
                if (cb != null) {
                    cb.run();
                }
                releaseResource();
                released = true;
                if (TRACK_GC_RELEASE && !manualCall) {
                    System.err.println("---\n" +
                            "Released buffer from:\n" +
                            initStackTrace + "Called from:\n" +
                            toString(new Exception()));
                }
            }
        }
    }

    protected abstract void releaseResource();

    protected abstract long getResourceId();

    public <R> R runBlockingConcurrentRelease(Callable<R> c) {
        synchronized (memObjLock) {
            if (!released) {
                try {
                    return c.call();
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            } else {
                throw new RuntimeException("Already disposed: " + getClass().getSimpleName() + ":" + toHexString(getResourceId()));
            }
        }
    }

    public void setCallback(Runnable callback) {
        this.callback = callback;
    }
}
