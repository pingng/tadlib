package com.codeberry.tadlib.provider.opencl.consts;

import com.sun.jna.ptr.IntByReference;

import javax.lang.model.element.UnknownElementException;
import javax.management.RuntimeErrorException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.function.Function;
import java.util.stream.Collectors;

import static java.util.Arrays.*;
import static java.util.function.Function.*;

public enum ErrorCode {
    /* Error Codes */
    CL_SUCCESS(0),
    CL_DEVICE_NOT_FOUND(-1),
    CL_DEVICE_NOT_AVAILABLE(-2),
    CL_COMPILER_NOT_AVAILABLE(-3),
    CL_MEM_OBJECT_ALLOCATION_FAILURE(-4),
    CL_OUT_OF_RESOURCES(-5),
    CL_OUT_OF_HOST_MEMORY(-6),
    CL_PROFILING_INFO_NOT_AVAILABLE(-7),
    CL_MEM_COPY_OVERLAP(-8),
    CL_IMAGE_FORMAT_MISMATCH(-9),
    CL_IMAGE_FORMAT_NOT_SUPPORTED(-10),
    CL_BUILD_PROGRAM_FAILURE(-11),
    CL_MAP_FAILURE(-12),
    CL_MISALIGNED_SUB_BUFFER_OFFSET(-13),
    CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST(-14),
    CL_INVALID_VALUE(-30),
    CL_INVALID_DEVICE_TYPE(-31),
    CL_INVALID_PLATFORM(-32),
    CL_INVALID_DEVICE(-33),
    CL_INVALID_CONTEXT(-34),
    CL_INVALID_QUEUE_PROPERTIES(-35),
    CL_INVALID_COMMAND_QUEUE(-36),
    CL_INVALID_HOST_PTR(-37),
    CL_INVALID_MEM_OBJECT(-38),
    CL_INVALID_IMAGE_FORMAT_DESCRIPTOR(-39),
    CL_INVALID_IMAGE_SIZE(-40),
    CL_INVALID_SAMPLER(-41),
    CL_INVALID_BINARY(-42),
    CL_INVALID_BUILD_OPTIONS(-43),
    CL_INVALID_PROGRAM(-44),
    CL_INVALID_PROGRAM_EXECUTABLE(-45),
    CL_INVALID_KERNEL_NAME(-46),
    CL_INVALID_KERNEL_DEFINITION(-47),
    CL_INVALID_KERNEL(-48),
    CL_INVALID_ARG_INDEX(-49),
    CL_INVALID_ARG_VALUE(-50),
    CL_INVALID_ARG_SIZE(-51),
    CL_INVALID_KERNEL_ARGS(-52),
    CL_INVALID_WORK_DIMENSION(-53),
    CL_INVALID_WORK_GROUP_SIZE(-54),
    CL_INVALID_WORK_ITEM_SIZE(-55),
    CL_INVALID_GLOBAL_OFFSET(-56),
    CL_INVALID_EVENT_WAIT_LIST(-57),
    CL_INVALID_EVENT(-58),
    CL_INVALID_OPERATION(-59),
    CL_INVALID_GL_OBJECT(-60),
    CL_INVALID_BUFFER_SIZE(-61),
    CL_INVALID_MIP_LEVEL(-62),
    CL_INVALID_GLOBAL_WORK_SIZE(-63),
    NVIDIA_ILLEGAL_READ_OR_WRITE_TO_A_BUFFER(-9999);

    private static final Map<Integer, ErrorCode> ERROR_CODES = Collections.unmodifiableMap(toMap());

    private static Map<Integer, ErrorCode> toMap() {
        return stream(ErrorCode.values())
                .collect(Collectors.toMap(
                        errorCode -> errorCode.code, identity(), (a, b) -> b));
    }

    public final int code;

    ErrorCode(int code) {
        this.code = code;
    }

    public static void throwOnError(Callable<Integer> callable) {
        throwOnError(callable, () -> null);
    }

    public static void throwOnError(Callable<Integer> callable, Callable<String> onErrorReturnMessage) {
        throwOnError(callable, onErrorReturnMessage, 3);
    }

    public static void throwOnError(Callable<Integer> callable, Callable<String> onErrorReturnMessage, int retryCount) {
        Integer errCode;
        ErrorCode code;
        long sleepMillis = 500;
        boolean shouldRetry;
        do {
            shouldRetry = false;
            try {
                errCode = callable.call();
            } catch (Exception e) {
                throw new OpenCLException(e);
            }
            code = errorOf(errCode);

            if (retryCount > 0 && code == CL_MEM_OBJECT_ALLOCATION_FAILURE) {
                shouldRetry = true;
                System.err.println("RETRY MEM: retriesLeft: " + retryCount + " sleep: " + sleepMillis);
                System.gc();
                try { Thread.sleep(sleepMillis); } catch (InterruptedException ignore) { }
            }

            retryCount--;
            sleepMillis *= 2;
        } while (shouldRetry);

        if (code != CL_SUCCESS) {
            String errorMsg;
            try {
                errorMsg = onErrorReturnMessage.call();
            } catch (Exception e) {
                errorMsg = "!!! While getting error message: " + e.toString();
            }
            throw new OpenCLException("Code: " + code.name() + " (" + errCode + ")" +
                    (errorMsg != null ? "\n" + errorMsg : ""));
        }
    }

    private static final ThreadLocal<IntByReference> THREAD_LOCAL_ERR_CODE_REF = ThreadLocal.withInitial(IntByReference::new);

    public static <R> R throwOnError(Function<IntByReference, R> clCallWithReturnAsRef) {
        IntByReference errCodeRef = THREAD_LOCAL_ERR_CODE_REF.get();

        R result = clCallWithReturnAsRef.apply(errCodeRef);
        ErrorCode code = errorOf(errCodeRef.getValue());
        if (code != CL_SUCCESS) {
            throw new OpenCLException("Code: " + code.name() + " (" + errCodeRef.getValue() + ")");
        }
        return result;
    }

    public static ErrorCode errorOf(int code) {
        ErrorCode c = ERROR_CODES.get(code);
        if (c != null) {
            return c;
        }
        throw new IllegalArgumentException("Unknown error code: " + code);
    }

    public static class OpenCLException extends RuntimeException {
        public OpenCLException(String msg) {
            super(msg);
        }

        public OpenCLException(Exception e) {
            super(e);
        }

        public OpenCLException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
