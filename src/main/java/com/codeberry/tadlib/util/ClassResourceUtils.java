package com.codeberry.tadlib.util;

import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;

public class ClassResourceUtils {
    public static String readString(Class<?> aClass, String resourceName) {
        try {
            URL url = aClass.getResource(resourceName);

            return Files.readString(Paths.get(url.toURI()));
        } catch (IOException | URISyntaxException e) {
            throw new RuntimeException(e);
        }
    }
}
