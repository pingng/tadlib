package com.codeberry.tadlib.util;

import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.annotation.PropertyAccessor;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectWriter;

public class StringUtils {
    static ObjectMapper objectMapper = new ObjectMapper();

    static {
        objectMapper.setVisibility(PropertyAccessor.FIELD, JsonAutoDetect.Visibility.ANY);
    }

    public static String toJson(Object toArray) {
        ObjectWriter
                w = objectMapper.writerWithDefaultPrettyPrinter();
        try {
            return w.writeValueAsString(toArray);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

}
