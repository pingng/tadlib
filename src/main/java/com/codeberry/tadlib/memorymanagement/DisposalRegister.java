package com.codeberry.tadlib.memorymanagement;

import java.lang.ref.Cleaner;
import java.util.*;
import java.util.concurrent.Callable;

import static java.util.Collections.emptyList;
import static java.util.Collections.singletonList;
import static java.util.Objects.*;
import static java.util.stream.Collectors.toList;

public class DisposalRegister {
    private static final Cleaner cleaner = Cleaner.create();

    private static final ThreadLocal<Stack<List<Disposable>>> CREATED_ARRAYS = ThreadLocal.withInitial(Stack::new);
    private static final ThreadLocal<List<Disposable>> TRAINING_ITERATION_DISPOSABLE = new ThreadLocal<>();

    public static void registerForDisposalAtEndOfModelIteration(Disposable disposable) {
        List<Disposable> disposables = TRAINING_ITERATION_DISPOSABLE.get();
        if (disposables != null) {
            disposables.add(disposable);
        }
    }

    /**
     * All additional resources created during the input callable will be disposed.
     *
     * This can be e.g., buffers for holding indices for max pool gradient operation.
     *
     * Resources returned by the callable is left untouched.
     */
    public static void modelIteration(Callable<List<Disposable>> callable) {
        synchronized (TRAINING_ITERATION_DISPOSABLE) {
            if (TRAINING_ITERATION_DISPOSABLE.get() != null) {
                throw new IllegalStateException("Cannot be called recursively");
            }

            List<Disposable> toBeDisposed = new ArrayList<>();
            List<Disposable> mustKeepAfterReturn = emptyList();
            try {
                TRAINING_ITERATION_DISPOSABLE.set(toBeDisposed);
                mustKeepAfterReturn = disposeAllExceptReturnedValues(callable);
            } finally {
                for (Disposable d : toBeDisposed) {
                    if (!mustKeepAfterReturn.contains(d)) {
                        d.dispose();
                    }
                }
                TRAINING_ITERATION_DISPOSABLE.set(null);
            }
        }
    }

    public static void registerDisposer(Object object, AbstractDisposer disposer) {
        Cleaner.Cleanable cleanable = cleaner.register(object, disposer::releaseByGcOrCleaner);

        disposer.setCleanable(cleanable);
    }

    public interface Disposable {
        void prepareDependenciesForDisposal();

        void dispose();
    }

    public static <A extends Disposable> void registerForDisposal(A d0, A d1) {
        registerForDisposal(d0);
        registerForDisposal(d1);
    }

    public static <A extends Disposable> A registerForDisposal(A arr) {
        if (arr != null) {
            Stack<List<Disposable>> listStack = CREATED_ARRAYS.get();

            if (!listStack.isEmpty()) {
                List<Disposable> list = listStack.peek();
                if (list != null) {
                    list.add(arr);
                }
            }
        }

        return arr;
    }

    @SuppressWarnings("unchecked")
    public static <R extends Disposable> R disposeAllExceptReturnedValue(Callable<R> callable) {
        List<Disposable> returnArrays = disposeAllExceptReturnedValues(() -> singletonList(callable.call()));

        return (R) returnArrays.get(0);
    }

    private static List<Disposable> disposeAllExceptReturnedValues(Callable<List<Disposable>> callable) {
        Stack<List<Disposable>> stack = CREATED_ARRAYS.get();

        List<Disposable> instantiatedDisposablesToKeep = new ArrayList<>();

        try {
            stack.push(new ArrayList<>(2048));

            List<Disposable> keepsFromCall = invokeAndEnsureNonNullElements(callable);
            for (Disposable d : keepsFromCall) {
                d.prepareDependenciesForDisposal();
            }

            List<Disposable> instantiatedDisposables = stack.peek();
            for (Disposable d : instantiatedDisposables) {
                if (keepsFromCall.contains(d)) {
                    instantiatedDisposablesToKeep.add(d);
                } else {
                    d.dispose();
                }
            }

            return keepsFromCall;
        } catch (Exception e) {
            throw new RuntimeException(e);
        } finally {
            stack.pop();

            if (!stack.isEmpty()) {
                List<Disposable> disposablesFromPrevStackCall = stack.peek();

                // Let the prev stack call handle disposal of these objects
                disposablesFromPrevStackCall.addAll(instantiatedDisposablesToKeep);
            }
        }
    }

    private static List<Disposable> invokeAndEnsureNonNullElements(Callable<List<Disposable>> callable) throws Exception {
        List<Disposable> fromCall = requireNonNullElseGet(callable.call(), Collections::emptyList);

        return fromCall.stream()
                .filter(Objects::nonNull)
                .collect(toList());
    }
}
