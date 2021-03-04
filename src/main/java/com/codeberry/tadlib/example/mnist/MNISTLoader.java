package com.codeberry.tadlib.example.mnist;

import com.codeberry.tadlib.example.TrainingData;
import com.codeberry.tadlib.tensor.Tensor;

import java.io.*;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import java.util.function.Function;
import java.util.zip.GZIPInputStream;

import static com.codeberry.tadlib.array.TArrayFactory.random;
import static com.codeberry.tadlib.array.TArrayFactory.randomInt;
import static com.codeberry.tadlib.provider.ProviderStore.*;
import static java.lang.Math.min;

public class MNISTLoader {
    public static final int IMAGE_SIZE = 28;
    public static final int OUTPUTS = 10;

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
        private String fileBasePath;
        private String downloadBasePath;
        private boolean downloadWhenMissing = true;
        private String trainingXFilename = "train-images-idx3-ubyte.gz";
        private String trainingYFilename = "train-labels-idx1-ubyte.gz";
        private String testXFilename = "t10k-images-idx3-ubyte.gz";
        private String testYFilename = "t10k-labels-idx1-ubyte.gz";

        public LoadParams() {
            loadRegularMNIST();
        }

        public static LoadParams params() {
            return new LoadParams();
        }

        public LoadParams loadRegularMNIST() {
            fileBasePath("./.data/mnist");
            downloadBasePath("http://yann.lecun.com/exdb/mnist");

            return this;
        }

        public LoadParams loadFashionMNIST() {
            fileBasePath("./.data/fashion");
            downloadBasePath("https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion");

            return this;
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

    private static Tensor loadOutput(InputStream inputStream, int maxExamples) throws IOException {
        DataInputStream in = new DataInputStream(new GZIPInputStream(inputStream));
        int magic = in.readInt();
        int images = min(in.readInt(), maxExamples);
        System.out.println("Load MNIST Output: Magic=" + magic + " OutputCountInFile=" + images);
        byte[] pixels = new byte[images];
        in.readFully(pixels);
        double[] data = new double[pixels.length];
        for (int i = 0; i < pixels.length; i++) {
            data[i] = pixels[i];
        }
        return new Tensor(array(data, shape(images, 1)),
                Tensor.GradientMode.NONE);
    }

    private static Tensor generateYTrain(Random rand, int examples) {
        return new Tensor(randomInt(rand, 0, 10, examples)
                .reshape(examples, 1), Tensor.GradientMode.NONE);
    }

    private static Tensor generateXTrain(Random rand, int examples) {
        int imageSize = 28;
        return new Tensor(random(rand, examples * imageSize * imageSize * 1)
                .reshape(examples, imageSize, imageSize, 1), Tensor.GradientMode.NONE);
    }

    private static Tensor loadInput(InputStream inputStream, int maxExamples) throws IOException {
        DataInputStream in = new DataInputStream(new GZIPInputStream(inputStream));
        int magic = in.readInt();
        int images = min(in.readInt(), maxExamples);
        int rows = in.readInt();
        int cols = in.readInt();
        System.out.println("Load MNIST Input: Magic=" + magic + " ImagesCountInFile=" + images + " Rows=" + rows + " Cols=" + cols);
        byte[] pixels = new byte[images * rows * cols];
        in.readFully(pixels);
        double[] data = new double[pixels.length];
        for (int i = 0; i < pixels.length; i++) {
            data[i] = (pixels[i] & 0xff) / 255.0;
        }
        return new Tensor(array(data, shape(images, rows, cols, 1)),
                Tensor.GradientMode.NONE);
    }
}
