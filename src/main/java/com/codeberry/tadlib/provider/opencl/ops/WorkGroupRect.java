package com.codeberry.tadlib.provider.opencl.ops;

class WorkGroupRect {
    final int height;
    final int width;

    private WorkGroupRect(int height, int width) {
        this.height = height;
        this.width = width;
    }

    @Override
    public String toString() {
        return "WorkGroupRect{" +
                "height=" + height +
                ", width=" + width +
                '}';
    }

    static WorkGroupRect evalMostSquareRect(int totalSize) {
        int width = (int) Math.sqrt(totalSize);
        while (width < totalSize && (totalSize % width) != 0) {
            width++;
        }
        return new WorkGroupRect(totalSize / width, width);
    }
}
