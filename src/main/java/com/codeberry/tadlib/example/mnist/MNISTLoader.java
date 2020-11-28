package com.codeberry.tadlib.example.mnist;

import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.array.TArray;
import com.codeberry.tadlib.example.TrainingData;
import com.codeberry.tadlib.tensor.Tensor;

import java.io.*;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import java.util.function.Function;
import java.util.zip.GZIPInputStream;

import static com.codeberry.tadlib.array.TArray.*;
import static java.lang.Math.min;

public class MNISTLoader {
    public static TrainingData generate(Random rand, int examples) {
        return new TrainingData(
                generateXTrain(rand, examples),
                generateYTrain(rand, examples),
                null, null);
    }

    public static TrainingData load(LoadParams params) {
        try {
            Tensor xTrain = loadFromFileOrUrl(params, params.trainingXFilename, params.trainingExamples, MNISTLoader::loadInput);
            Tensor yTrain = loadFromFileOrUrl(params, params.trainingYFilename, params.trainingExamples, MNISTLoader::loadOutput);
            Tensor xTest = loadFromFileOrUrl(params, params.testXFilename, params.testExamples, MNISTLoader::loadInput);
            Tensor yTest = loadFromFileOrUrl(params, params.testYFilename, params.testExamples, MNISTLoader::loadOutput);

            return new TrainingData(xTrain, yTrain, xTest, yTest);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private interface TensorLoader {
        Tensor load(InputStream inputStream, int examplesToLoa) throws IOException;
    }

    private static Tensor loadFromFileOrUrl(LoadParams params, String filename, int examples, TensorLoader loader) throws IOException {
        if (examples == 0) {
            return null;
        }

        try (InputStream in = openOrDownloadFile(params.fileBasePath, filename, urlStreamSupport(params))) {
            return loader.load(in, examples);
        }
    }

    private static Function<String, InputStream> urlStreamSupport(LoadParams params) {
        return params.downloadWhenMissing ? urlStream(params.downloadBasePath) : throwMissingFile();
    }

    private static InputStream openOrDownloadFile(String basePath, String filename, Function<String, InputStream> urlStreamOpener) {
        Path path = Path.of(basePath, filename);
        if (Files.exists(path)) {
            try {
                return new FileInputStream(path.toFile());
            } catch (FileNotFoundException e) {
                throw new RuntimeException("should not happen", e);
            }
        }
        try (InputStream urlStream = urlStreamOpener.apply(filename)) {
            Files.createDirectories(path.getParent());
            Files.copy(urlStream, path);

            return new FileInputStream(path.toFile());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static Function<String, InputStream> urlStream(String downloadBase) {
        return filename -> {
            String base = downloadBase + (downloadBase.endsWith("/") ? "" : "/");
            try {
                URL url = new URL(base + filename);
                return url.openStream();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        };
    }

    private static Function<String, InputStream> throwMissingFile() {
        return filename -> {
            throw new RuntimeException("Could not find file: " + filename);
        };
    }

    public static class LoadParams {
        private int trainingExamples;
        private int testExamples;
        private String fileBasePath = "./.data/mnist";
        private String downloadBasePath = "http://yann.lecun.com/exdb/mnist";
        private boolean downloadWhenMissing = true;
        private String trainingXFilename = "train-images-idx3-ubyte.gz";
        private String trainingYFilename = "train-labels-idx1-ubyte.gz";
        private String testXFilename = "t10k-images-idx3-ubyte.gz";
        private String testYFilename = "t10k-labels-idx1-ubyte.gz";

        public static LoadParams params() {
            return new LoadParams();
        }

        public LoadParams trainingExamples(int trainingExamples) {
            this.trainingExamples = trainingExamples;
            return this;
        }

        public LoadParams testExamples(int testExamples) {
            this.testExamples = testExamples;
            return this;
        }

        public LoadParams fileBasePath(String fileBasePath) {
            this.fileBasePath = fileBasePath;
            return this;
        }

        public LoadParams downloadBasePath(String downloadBasePath) {
            this.downloadBasePath = downloadBasePath;
            return this;
        }

        public LoadParams downloadWhenMissing(boolean downloadWhenMissing) {
            this.downloadWhenMissing = downloadWhenMissing;
            return this;
        }

        public LoadParams trainingXFilename(String trainingXFilename) {
            this.trainingXFilename = trainingXFilename;
            return this;
        }

        public LoadParams trainingYFilename(String trainingYFilename) {
            this.trainingYFilename = trainingYFilename;
            return this;
        }

        public LoadParams testingXFilename(String testingXFilename) {
            this.testXFilename = testingXFilename;
            return this;
        }

        public LoadParams testingYFilename(String testingYFilename) {
            this.testYFilename = testingYFilename;
            return this;
        }
    }

    public static Tensor toOneHot(Tensor yTrain) {
        int examples = yTrain.getShape().at(0);
        TArray out = new TArray(new double[examples][10]);
        int[] indices = out.shape.newIndexArray();
        for (int i = 0; i < examples; i++) {
            indices[0] = i;
            indices[1] = (int) yTrain.dataAt(i, 0);
            out.setAt(indices, 1.0);
        }
        return new Tensor(out, Tensor.GradientMode.NONE);
    }

    private static Tensor loadOutput(InputStream inputStream, int maxExamples) throws IOException {
        DataInputStream in = new DataInputStream(new GZIPInputStream(inputStream));
        int magic = in.readInt();
        int images = min(in.readInt(), maxExamples);
        System.out.println("Magic: " + magic);
        System.out.println("Number of images: " + images);
        byte[] pixels = new byte[images];
        in.readFully(pixels);
        double[] data = new double[pixels.length];
        for (int i = 0; i < pixels.length; i++) {
            data[i] = pixels[i];
        }
        return new Tensor(new TArray(data, new Shape(images, 1)),
                Tensor.GradientMode.NONE);
    }

    private static Tensor generateYTrain(Random rand, int examples) {
        return new Tensor(randMatrixInt(rand, 0, 10, examples)
                .reshape(examples, 1), Tensor.GradientMode.NONE);
    }

    private static Tensor generateXTrain(Random rand, int examples) {
        int imageSize = 28;
        return new Tensor(rand(rand, examples * imageSize * imageSize * 1)
                .reshape(examples, imageSize, imageSize, 1), Tensor.GradientMode.NONE);
    }

    private static Tensor loadInput(InputStream inputStream, int maxExamples) throws IOException {
        DataInputStream in = new DataInputStream(new GZIPInputStream(inputStream));
        int magic = in.readInt();
        int images = min(in.readInt(), maxExamples);
        int rows = in.readInt();
        int cols = in.readInt();
        System.out.println("Magic: " + magic);
        System.out.println("Number of images: " + images);
        System.out.println("Rows: " + rows);
        System.out.println("Cols: " + cols);
        byte[] pixels = new byte[images * rows * cols];
        in.readFully(pixels);
        double[] data = new double[pixels.length];
        for (int i = 0; i < pixels.length; i++) {
            data[i] = (pixels[i] & 0xff) / 255.0;
        }
        return new Tensor(new TArray(data, new Shape(images, rows, cols, 1)),
                Tensor.GradientMode.NONE);
    }
}
