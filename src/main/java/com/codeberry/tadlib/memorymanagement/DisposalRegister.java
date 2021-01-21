package com.codeberry.tadlib.memorymanagement;

import java.lang.ref.Cleaner;
import java.util.ArrayList;
import java.util.List;
import java.util.Stack;
import java.util.concurrent.Callable;

import static java.util.Collections.singletonList;

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
    public static <T extends DisposableContainer<? extends Disposable>> void modelIteration(Callable<List<T>> callable) {
        synchronized (TRAINING_ITERATION_DISPOSABLE) {
            if (TRAINING_ITERATION_DISPOSABLE.get() != null) {
                throw new IllegalStateException("Cannot be called recursively");
            }

            List<Disposable> disposables = new ArrayList<>();
            List<Disposable> keeps = new ArrayList<>();
            try {
                TRAINING_ITERATION_DISPOSABLE.set(disposables);
                keeps.addAll(disposeAllExceptContainedReturnValues(callable));
            } finally {
                for (Disposable d : disposables) {
                    if (!keeps.contains(d)) {
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

    public interface DisposableContainer<R extends Disposable> {
        List<R> getDisposables();
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

    /**
     *
     */
    @SuppressWarnings("unchecked")
    public static <T extends DisposableContainer<R>, R extends Disposable> R disposeAllExceptReturnedValues(Callable<T> callable) {
        List<Disposable> returnArrays = disposeAllExceptContainedReturnValues(() -> singletonList(callable.call()));

        return (R) returnArrays.get(0);
    }

    public static <T extends DisposableContainer<? extends Disposable>> List<Disposable> disposeAllExceptContainedReturnValues(Callable<List<T>> callable) {
        Stack<List<Disposable>> stack = CREATED_ARRAYS.get();

        List<Disposable> instantiatedDisposablesToKeep = new ArrayList<>();

        try {
            stack.push(new ArrayList<>());

            List<Disposable> keepsFromCall = new ArrayList<>();
            List<T> returnedContainer = callable.call();
            for (T c : returnedContainer) {
                if (c instanceof Disposable) {
                    Disposable nda = (Disposable) c;

                    nda.prepareDependenciesForDisposal();
                    keepsFromCall.add(nda);
                } else if (c != null) {
                    List<? extends Disposable> disposables = c.getDisposables();
                    for (Disposable d : disposables) {
                        if (d != null) {
                            d.prepareDependenciesForDisposal();
                            keepsFromCall.add(d);
                        }
                    }
                }
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
}
