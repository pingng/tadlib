package com.codeberry.tadlib.array;

import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.java.NDArray;
import com.codeberry.tadlib.provider.java.JavaProvider;
import com.codeberry.tadlib.provider.java.JavaShape;
//import com.codeberry.tadlib.provider.opencl.OpenCLProvider;
import com.codeberry.tadlib.tensor.conv2ddata.Conv2DData_2in_1out;
import com.codeberry.tadlib.tensor.conv2ddata.Conv2DData_2in_2out;
import com.codeberry.tadlib.tensor.conv2ddata.Conv2DData_3x3_2x2_Filter;
import com.codeberry.tadlib.tensor.conv2ddata.Conv2DExample;
import com.codeberry.tadlib.util.MatrixTestUtils;
import com.codeberry.tadlib.util.StringUtils;
import com.google.gson.Gson;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Random;

import static com.codeberry.tadlib.array.TArrayFactory.*;
import static com.codeberry.tadlib.util.MatrixTestUtils.assertEqualsMatrix;
import static com.codeberry.tadlib.util.StringUtils.toJson;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Arrays.deepEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TArrayConv2D {
    @BeforeEach
    public void init() {
//        ProviderStore.setProvider(new JavaProvider()); enableMultiThreading();
        ProviderStore.setProvider(new JavaProvider());
    }

    // Filer=2 input=3 chanIn=1 chanOut=1 FAIL
    @Test
    @Disabled public void testMethod5() {
        NDArray a = ProviderStore.array(new double[] {
                5, 2, 4,
                77,11,44,
                13,17,23
        }).reshape(1, 3, 3, 1);
        NDArray gradient = ProviderStore.array(new double[] {
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
        }).reshape(3, 3, 1, 1);
        NDArray filter = ProviderStore.array(new double[] {
                1, 1, 1, 1
        }).reshape(2, 2, 1, 1);

        NDArray y = a.conv2d(gradient, 0, 0, 2, 2);
        NDArray realY = a.conv2d(filter);

        System.out.println(toJson(y.toDoubles(), StringUtils.JsonPrintMode.COMPACT));
        System.out.println(toJson(realY.toDoubles(), StringUtils.JsonPrintMode.COMPACT));
        System.out.println((
                5*1 +  2*2 + 4*3+
                77*4 + 11*5 +44*6+
                13*7 + 17*8 +23*9));
    }

    @Test
    public void testMethod() {
        NDArray a = ProviderStore.array(Conv2DData_3x3_2x2_Filter.input);
        NDArray b = ProviderStore.array(Conv2DData_3x3_2x2_Filter.filter);

        NDArray y = a.conv2d(b);

        System.out.println(toJson(y.toDoubles(), StringUtils.JsonPrintMode.COMPACT));
        System.out.println(toJson(Conv2DData_3x3_2x2_Filter.y, StringUtils.JsonPrintMode.COMPACT));

        assertTrue(deepEquals(Conv2DData_3x3_2x2_Filter.y,
                (Object[]) y.toDoubles()));

    }

    @Test
    public void testMethod2() {
        NDArray a = ProviderStore.array(Conv2DData_2in_1out.input);
        NDArray b = ProviderStore.array(Conv2DData_2in_1out.filter);

        NDArray y = a.conv2d(b);
        System.out.println(toJson(y.toDoubles(), StringUtils.JsonPrintMode.COMPACT));
        System.out.println(toJson(Conv2DData_2in_1out.y, StringUtils.JsonPrintMode.COMPACT));
        assertTrue(deepEquals(Conv2DData_2in_1out.y,
                (Object[]) y.toDoubles()));

    }

    @Test
    public void testMethod3() {
        NDArray a = ProviderStore.array(Conv2DData_2in_2out.input);
        NDArray b = ProviderStore.array(Conv2DData_2in_2out.filter);

        NDArray y = a.conv2d(b);
        System.out.println(toJson(y.toDoubles(), StringUtils.JsonPrintMode.COMPACT));
        System.out.println(toJson(Conv2DData_2in_2out.y, StringUtils.JsonPrintMode.COMPACT));
        assertTrue(deepEquals(Conv2DData_2in_2out.y,
                (Object[]) y.toDoubles()));

    }

    @Disabled
    @Test
    public void testMethod4() {
        //NDArray a = ProviderStore.array(new double[16][28][28][48]);
        //NDArray b = ProviderStore.array(new double[11][11][48][48]);
        //NDArray a = ProviderStore.array(new double[16][28][28][32]);
        //NDArray b = ProviderStore.array(new double[7][7][32][64]);
        Random rand = new Random(3);
        //MultiThreadingSupport.enableMultiThreading();
        JavaShape jas = JavaShape.shape(16, 32, 32, 64);
        JavaShape jbs = JavaShape.shape(11, 11, 64, 128);
        NDArray ja = new NDArray(randomDoubles(rand, jas.size)).reshape(jas);
        NDArray jb = new NDArray(randomDoubles(rand, jbs.size)).reshape(jbs);

        NDArray a = ProviderStore.array((double[][][][]) ja.toDoubles());
        NDArray b = ProviderStore.array((double[][][][]) jb.toDoubles());

        a.waitForValueReady();
        b.waitForValueReady();
        // ----
        long st = System.currentTimeMillis();
        NDArray y = a.conv2d(b);
        y.waitForValueReady();
        long used = System.currentTimeMillis() - st;
        System.out.println("used = " + used);
        Object ndResult = y.toDoubles();

        // ----
        long jst = System.currentTimeMillis();
        NDArray jy = (NDArray) ja.conv2d(jb);
        long jused = System.currentTimeMillis() - jst;
        System.out.println("jused = " + jused);
        System.out.println("Diff: " + (jused / used));
        Object javaResult = jy.toDoubles();

        System.out.println(y.shape);
        assertEquals(jy.shape, y.shape);
//        System.out.println(StringUtils.toJson(y.toDoubles(), StringUtils.JsonPrintMode.COMPACT));
//        System.out.println(StringUtils.toJson(jy.toDoubles(), StringUtils.JsonPrintMode.COMPACT));
        MatrixTestUtils.assertEqualsMatrix(javaResult, ndResult);


    }

    @Test
    public void testFromFile() {
        Gson gson = new Gson();
        InputStream in = getClass().getResourceAsStream("/com/codeberry/tadlib/tensor/conv2ddata/conv2d_test_data_rnd.json");
        Conv2DExample[] list = gson.fromJson(new InputStreamReader(in, UTF_8),
                Conv2DExample[].class);
        for (Conv2DExample ex : list) {
            ex.convertListsToArrays();
        }
        for (Conv2DExample ex : list) {
            testExample(ex);
        }
    }

    private void testExample(Conv2DExample ex) {
        System.out.println(ex.config.name);

        NDArray input = ProviderStore.array((double[][][][]) ex.input);
        NDArray filter = ProviderStore.array((double[][][][]) ex.filter);

        NDArray y = input.conv2d(filter);
        System.out.println(toJson(y.toDoubles(), StringUtils.JsonPrintMode.COMPACT));
        System.out.println(toJson(ex.y, StringUtils.JsonPrintMode.COMPACT));
        MatrixTestUtils.assertEqualsMatrix(ex.y, y.toDoubles());
    }

}
