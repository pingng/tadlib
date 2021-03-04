package com.codeberry.tadlib.memorymanagement;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.lang.ref.Cleaner;
import java.util.concurrent.Callable;

import static java.lang.Long.toHexString;

public abstract class AbstractDisposer {
    // TODO: allow to be set at runtime
    private static final boolean TRACK_GC_RELEASE = false;

    // The same lock object is used by all views
    protected final Object memObjLock;
    protected boolean released;
    protected Cleaner.Cleanable cleanable;
    protected boolean releasedManually;

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

    public void release() {
        synchronized (memObjLock) {
            if (cleanable != null) {
                releasedManually = true;
                cleanable.clean();
            }
        }
    }

    void setCleanable(Cleaner.Cleanable cleanable) {
        synchronized (memObjLock) {
            this.cleanable = cleanable;
        }
    }

    void releaseByGcOrCleaner() {
        synchronized (memObjLock) {
            if (!released) {
                if (this.callback != null) {
                    this.callback.run();
                }
                releaseResource();
                cleanable = null;
                released = true;
                if (TRACK_GC_RELEASE && !releasedManually) {
                    System.err.println("---\n" +
                            "Disposed object with instantiation stacktrace:\n" +
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
        synchronized (memObjLock) {
            this.callback = callback;
        }
    }

    public boolean isReleased() {
        synchronized (memObjLock) {
            return released;
        }
    }
}
