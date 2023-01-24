package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.tensor.conv2ddata.*;
import com.google.gson.Gson;
import org.junit.jupiter.api.Test;

import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import static com.codeberry.tadlib.provider.ProviderStore.array;
import static com.codeberry.tadlib.util.MatrixTestUtils.assertEqualsMatrix;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Arrays.deepEquals;
import static java.util.Arrays.deepToString;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TensorConv2DTest {

    // TODO: test in/out channels

    @Test
    public void testFromFile() {
        Gson gson = new Gson();
        InputStream in = getClass().getResourceAsStream("/com/codeberry/tadlib/tensor/conv2ddata/conv2d_test_data_rnd.json");
        Conv2DExample[] list = gson.fromJson(new InputStreamReader(in, UTF_8),
                Conv2DExample[].class);
        for (Conv2DExample ex : list) {
            ex.convertListsToArrays();
        }
        boolean failed = false;
        List<Boolean> failedExamples = new ArrayList<>();
        for (Conv2DExample ex : list) {
            try {
                testExample(ex);
                failedExamples.add(true);
            } catch (AssertionError e) {
                failedExamples.add(false);
                e.printStackTrace();
                failed = true;
            }
        }

        for (int i = 0; i < list.length; i++) {
            Conv2DExample.Config cfg = list[i].config;
            System.out.println("Filer=" + cfg.filter_size + " input=" + cfg.input_size +
                    " chanIn=" + cfg.input_channels +
                    " chanOut=" + cfg.output_channels + (!failedExamples.get(i) ? " FAIL" : ""));
        }
        assertFalse(failed);
    }

    private static void testExample(Conv2DExample ex) {
        System.out.println("'" + ex.config.name + "'");

        Tensor input = new Tensor((double[][][][]) ex.input);
        Tensor filter = new Tensor((double[][][][]) ex.filter);

        Tensor y = Ops.conv2d(input, filter);
        y.backward(array((double[][][][]) ex.grad_y));

        //System.out.println(deepToString(Conv2DData_3x3_2x2_Filter.y));
        //System.out.println(deepToString((Object[]) y.m.toArray()));
        assertEqualsMatrix(ex.y, y.val().toDoubles());

        //System.out.println(deepToString((Object[]) ex.grad_input));
        //System.out.println(deepToString((Object[]) input.gradient.m.toArray()));
        assertEqualsMatrix(ex.grad_input, input.grad().toDoubles());

        //System.out.println(deepToString(Conv2DData_3x3_2x2_Filter.grad_filter));
        //System.out.println(deepToString((Object[]) filter.gradient.m.toArray()));
        assertEqualsMatrix(ex.grad_filter, filter.grad().toDoubles());
    }

    @Test
    public void conv2D_Simple() {
        Tensor input = new Tensor(Conv2DData_3x3_2x2_Filter.input);
        Tensor filter = new Tensor(Conv2DData_3x3_2x2_Filter.filter);

        Tensor y = Ops.conv2d(input, filter);
        y.backward(array(Conv2DData_3x3_2x2_Filter.grad_y));

        //System.out.println(deepToString(Conv2DData_3x3_2x2_Filter.y));
        //System.out.println(deepToString((Object[]) y.m.toArray()));
        assertEqualsMatrix(Conv2DData_3x3_2x2_Filter.y,
                y.val().toDoubles());

        //System.out.println(deepToString(Conv2DData_3x3_2x2_Filter.grad_filter));
        //System.out.println(deepToString((Object[]) filter.gradient.m.toArray()));
        assertEqualsMatrix(Conv2DData_3x3_2x2_Filter.grad_filter,
                filter.grad().toDoubles());

        System.out.println(deepToString(Conv2DData_3x3_2x2_Filter.grad_input));
        System.out.println(deepToString((Object[]) input.grad().toDoubles()));
        assertEqualsMatrix(Conv2DData_3x3_2x2_Filter.grad_input,
                input.grad().toDoubles());

    }

    @Test
    public void conv2D() {
        Tensor input = new Tensor(Conv2DData_1_Filter.input);
        Tensor filter = new Tensor(Conv2DData_1_Filter.filter);

        Tensor y = Ops.conv2d(input, filter);
        y.backward(array(Conv2DData_1_Filter.grad_y));

        assertTrue(deepEquals(Conv2DData_1_Filter.y,
                (Object[]) y.val().toDoubles()));

        System.out.println(deepToString(Conv2DData_1_Filter.grad_filter));
        System.out.println(deepToString((Object[]) filter.grad().toDoubles()));
        assertTrue(deepEquals(Conv2DData_1_Filter.grad_filter,
                (Object[]) filter.grad().toDoubles()));

        //System.out.println(deepToString(Conv2DData_1_Filter.grad_input));
        //System.out.println(deepToString((Object[]) input.gradient.m.toArray()));
        assertTrue(deepEquals(Conv2DData_1_Filter.grad_input,
                (Object[]) input.grad().toDoubles()));

    }

    @Test
    public void conv2D2_1() {
        Tensor input = new Tensor(Conv2DData_2in_1out.input);
        Tensor filter = new Tensor(Conv2DData_2in_1out.filter);

        Tensor y = Ops.conv2d(input, filter);
        y.backward(array(Conv2DData_2in_1out.grad_y));

        //System.out.println(deepToString(Conv2DData_2in_1out.y));
        //System.out.println(deepToString((Object[]) y.m.toArray()));
        assertTrue(deepEquals(Conv2DData_2in_1out.y,
                (Object[]) y.val().toDoubles()));

        System.out.println(deepToString(Conv2DData_2in_1out.grad_input));
        System.out.println(deepToString((Object[]) filter.grad().toDoubles()));
        assertTrue(deepEquals(Conv2DData_2in_1out.grad_filter,
                (Object[]) filter.grad().toDoubles()));

        //System.out.println(deepToString(Conv2DData_2in_1out.grad_input));
        //System.out.println(deepToString((Object[]) input.gradient.m.toArray()));
        assertTrue(deepEquals(Conv2DData_2in_1out.grad_input,
                (Object[]) input.grad().toDoubles()));
    }

    @Test
    public void conv2D2() {
        Tensor input = new Tensor(Conv2DData_2in_2out.input);
        Tensor filter = new Tensor(Conv2DData_2in_2out.filter);

        Tensor y = Ops.conv2d(input, filter);
        y.backward(array(Conv2DData_2in_2out.grad_y));

        assertTrue(deepEquals(Conv2DData_2in_2out.y,
                (Object[]) y.val().toDoubles()));

        //System.out.println(deepToString(Conv2DData_3x3_2x2_Filter.grad_filter));
        //System.out.println(deepToString((Object[]) filter.gradient.m.toArray()));
        assertTrue(deepEquals(Conv2DData_2in_2out.grad_filter,
                (Object[]) filter.grad().toDoubles()));

        System.out.println(deepToString(Conv2DData_2in_2out.grad_input));
        System.out.println(deepToString((Object[]) input.grad().toDoubles()));
        assertTrue(deepEquals(Conv2DData_2in_2out.grad_input,
                (Object[]) input.grad().toDoubles()));
    }

    @Test
    public void conv2D_2x2_4x4() {
        Tensor input = new Tensor(Conv2DData_2x2_4x4.input);
        Tensor filter = new Tensor(Conv2DData_2x2_4x4.filter);

        Tensor y = Ops.conv2d(input, filter);
        y.backward(array(Conv2DData_2x2_4x4.grad_y));

        assertTrue(deepEquals(Conv2DData_2x2_4x4.y,
                (Object[]) y.val().toDoubles()));

        System.out.println(deepToString(Conv2DData_2x2_4x4.grad_filter));
        System.out.println(deepToString((Object[]) filter.grad().toDoubles()));
        assertTrue(deepEquals(Conv2DData_2x2_4x4.grad_filter,
                (Object[]) filter.grad().toDoubles()));

        System.out.println(deepToString(Conv2DData_2x2_4x4.grad_input));
        System.out.println(deepToString((Object[]) input.grad().toDoubles()));
        assertTrue(deepEquals(Conv2DData_2x2_4x4.grad_input,
                (Object[]) input.grad().toDoubles()));
    }
}
