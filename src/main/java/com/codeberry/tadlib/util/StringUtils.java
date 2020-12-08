package com.codeberry.tadlib.util;


import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.time.temporal.Temporal;
import java.util.*;

import static java.util.AbstractMap.Entry;
import static java.util.AbstractMap.SimpleEntry;
import static java.util.stream.Collectors.toMap;

public class StringUtils {
    private static final Set<Class<?>> OUTPUT_CLASSES = Set.of(Boolean.class, Byte.class, Short.class,
            Character.class, Integer.class, Long.class, Float.class, Double.class, Temporal.class,
            String.class);
    private static final Set<Class<?>> NO_QUOTE_CLASSES = Set.of(
            Boolean.class, Byte.class, Short.class, Character.class, Integer.class, Long.class, Float.class, Double.class,
            boolean.class, byte.class, short.class, char.class, int.class, long.class, float.class, double.class);

    public static String toJson(Object value) {
        return toJson(value, JsonPrintMode.PRETTY);
    }

    public static String toJson(Object value, JsonPrintMode mode) {
        Object fieldValues = asMapOrValue(value);

        return renderString(fieldValues, mode);
    }

    private static String renderString(Object fieldValues, JsonPrintMode mode) {
        StringBuilder buf = new StringBuilder();

        renderString(fieldValues, mode, 0, buf);

        return buf.toString();
    }

    @SuppressWarnings("unchecked")
    private static void renderString(Object value, JsonPrintMode mode, int indent, StringBuilder buf) {
        if (value instanceof Map) {
            renderMap((Map<String, Object>) value, mode, indent, buf);
        } else if(value instanceof Object[]) {
            renderArray((Object[]) value, mode, indent, buf);
        } else {
            buf.append(value);
        }
    }

    private static void renderMap(Map<String, Object> map, JsonPrintMode mode, int indent, StringBuilder buf) {
        buf.append("{").append(mode.newLine);
        for (Iterator<Entry<String, Object>> iterator = map.entrySet().iterator(); iterator.hasNext(); ) {
            Entry<String, Object> e = iterator.next();
            buf.append(mode.indent.repeat(indent + 1))
                    .append('"')
                    .append(e.getKey())
                    .append("\" : ");
            renderString(e.getValue(), mode, indent + 1, buf);
            if (iterator.hasNext()) {
                buf.append(mode.comma);
                buf.append(mode.newLine);
            }
        }
        buf.append(mode.newLine);
        buf.append(mode.indent.repeat(indent)).append("}");
    }

    private static void renderArray(Object[] array, JsonPrintMode mode, int indent, StringBuilder buf) {
        buf.append("[").append(mode.newLine);
        for (int i = 0, arrLength = array.length; i < arrLength; i++) {
            buf.append(mode.indent.repeat(indent + 1));
            renderString(array[i], mode, indent + 1, buf);
            if (i < arrLength - 1) {
                buf.append(mode.comma);
                buf.append(mode.newLine);
            }
        }
        buf.append(mode.newLine);
        buf.append(mode.indent.repeat(indent)).append("]");
    }

    private static Object asMapOrValue(Object value) {
        Class<?> valueClass = value.getClass();
        if (OUTPUT_CLASSES.contains(valueClass) || valueClass.isPrimitive()) {
            if (NO_QUOTE_CLASSES.contains(valueClass)) {
                return value.toString();
            }
            return '"' + value.toString() + '"';
        }
        if (Enum.class.isAssignableFrom(valueClass)) {
            return '"' + value.toString() + '"';
        }

        if (value instanceof List) {
            List<?> l = (List<?>) value;
            Object[] out = new Object[l.size()];
            for (int i = 0; i < l.size(); i++) {
                out[i] = asMapOrValue(l.get(i));
            }
            return out;
        }
        if (valueClass.isArray()) {
            int len = Array.getLength(value);
            Object[] out = new Object[len];
            for (int i = 0; i < len; i++) {
                out[i] = asMapOrValue(Array.get(value, i));
            }
            return out;
        }

        Field[] fields = valueClass.getDeclaredFields();

        Entry<String, Object>[] entries = toEntries(fields, value);
        return Arrays.stream(entries)
                .collect(toMap(Entry::getKey, Entry::getValue, (old, newVal) -> newVal, LinkedHashMap::new));
    }

    @SuppressWarnings("unchecked")
    private static Entry<String, Object>[] toEntries(Field[] fields, Object value) {
        return Arrays.stream(fields)
                .map(f -> {
                    f.setAccessible(true);
                    String name = f.getName();
                    try {
                        Object rawValue = f.get(value);
                        if (rawValue != null) {
                            Object mapValue = asMapOrValue(rawValue);
                            if (mapValue != null) {
                                return new SimpleEntry<>(name, mapValue);
                            }
                        }
                    } catch (IllegalAccessException e) {
                        e.printStackTrace();
                    }
                    return null;
                })
                .filter(Objects::nonNull)
                .toArray(Entry[]::new);
    }

    public enum JsonPrintMode {
        PRETTY("  ", ",", "\n"),
        COMPACT("", ", ", ""),
        ;

        private final String indent;
        private final String comma;
        private final String newLine;

        JsonPrintMode(String indent, String comma, String newLine) {
            this.indent = indent;
            this.comma = comma;
            this.newLine = newLine;
        }
    }

}
