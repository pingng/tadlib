package com.codeberry.tadlib.provider.opencl.platform;

import com.sun.jna.Pointer;

public class Platform extends Pointer {
    private final Info info;
    private final PlatformsPointer parent;

    public Platform(int index, Info info, PlatformsPointer parent) {
        super(Pointer.nativeValue(parent.getPlatFormPointer(index)));
        this.info = info;
        this.parent = parent;
    }

    public Info getInfo() {
        return info;
    }

    @Override
    public String toString() {
        return "Platform{" +
                "info=" + info +
                '}';
    }

    public static Platform fromPlatformsIndex(PlatformsPointer parent, int index) {
        Info info = parent.getPlatformInfo(parent, index);
        return new Platform(index, info, parent);
    }

    static class Info {
        public final String name;
        public final String version;

        Info(String name, String version) {
            this.name = name;
            this.version = version;
        }

        @Override
        public String toString() {
            return "{name=" + name + ", version=" + version + "}";
        }
    }
}
