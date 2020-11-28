package com.codeberry.tadlib.util;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

public class ReflectionUtils {
    @SuppressWarnings("unchecked")
    public static <F> void copyFieldOfClass(Class<F> fieldClass,
                                            Object src, Object target,
                                            Function<F, F> valueCopier) {
        Class<?> srcClass = src.getClass();
        if (srcClass.isAssignableFrom(target.getClass())) {
            Field[] fields = srcClass.getDeclaredFields();
            for (Field f : fields) {
                if (f.getType() == fieldClass) {
                    f.setAccessible(true);
                    try {
                        Object org = f.get(src);
                        if (org != null) {
                            Object newValue = valueCopier.apply((F) org);
                            f.set(target, newValue);
                        }
                    } catch (IllegalAccessException e) {
                        throw new RuntimeException(e);
                    }
                }
            }
        } else {
            throw new IllegalArgumentException("Class of source and target is incompatible" +
                    srcClass + " vs " + target.getClass());
        }
    }

    @SuppressWarnings("unchecked")
    public static <F> List<F> getFieldValues(Class<F> fieldClass, Object src) {
        Class<?> srcClass = src.getClass();
        Field[] fields = srcClass.getDeclaredFields();
        List<F> ret = new ArrayList<>();
        for (Field f : fields) {
            if (f.getType() == fieldClass) {
                f.setAccessible(true);
                try {
                    F org = (F) f.get(src);
                    if (org != null) {
                        ret.add(org);
                    }
                } catch (IllegalAccessException e) {
                    throw new RuntimeException(e);
                }
            }
        }
        return ret;
    }
}
