package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.array.TArray;
import com.codeberry.tadlib.tensor.conv2ddata.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import org.junit.jupiter.api.Test;

import java.io.IOException;

import static com.codeberry.tadlib.tensor.MatrixTestUtils.assertEqualsMatrix;
import static java.util.Arrays.deepEquals;
import static java.util.Arrays.deepToString;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TensorConv2DTest {
    @Test
    public void testFromFile() throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        ObjectReader r = mapper.reader(Conv2DExample[].class);
        Conv2DExample[] list = r.readValue(getClass().getResourceAsStream("/com/codeberry/tadlib/tensor/conv2ddata/conv2d_test_data_rnd.json"));
        for (Conv2DExample ex : list) {
            ex.convertListsToArrays();
        }
        for (Conv2DExample ex : list) {
            testExample(ex);
        }
    }

    private void testExample(Conv2DExample ex) {
        System.out.println(ex.config.name);

        Tensor input = new Tensor((double[][][][]) ex.input);
        Tensor filter = new Tensor((double[][][][]) ex.filter);

        Tensor y = Ops.conv2d(input, filter);
        y.backward(new TArray((double[][][][]) ex.grad_y));

        //System.out.println(deepToString(Conv2DData_3x3_2x2_Filter.y));
        //System.out.println(deepToString((Object[]) y.m.toArray()));
        assertEqualsMatrix(ex.y, y.vals.toDoubles());

        //System.out.println(deepToString(Conv2DData_3x3_2x2_Filter.grad_filter));
        //System.out.println(deepToString((Object[]) filter.gradient.m.toArray()));
        assertEqualsMatrix(ex.grad_filter, filter.gradient.toDoubles());

        //System.out.println(deepToString((Object[]) ex.grad_input));
        //System.out.println(deepToString((Object[]) input.gradient.m.toArray()));
        assertEqualsMatrix(ex.grad_input, input.gradient.toDoubles());

    }
    @Test
    public void conv2D_Simple() {
        Tensor input = new Tensor(Conv2DData_3x3_2x2_Filter.input);
        Tensor filter = new Tensor(Conv2DData_3x3_2x2_Filter.filter);

        Tensor y = Ops.conv2d(input, filter);
        y.backward(new TArray(Conv2DData_3x3_2x2_Filter.grad_y));

        //System.out.println(deepToString(Conv2DData_3x3_2x2_Filter.y));
        //System.out.println(deepToString((Object[]) y.m.toArray()));
        assertTrue(deepEquals(Conv2DData_3x3_2x2_Filter.y,
                (Object[]) y.vals.toDoubles()));

        //System.out.println(deepToString(Conv2DData_3x3_2x2_Filter.grad_filter));
        //System.out.println(deepToString((Object[]) filter.gradient.m.toArray()));
        assertTrue(deepEquals(Conv2DData_3x3_2x2_Filter.grad_filter,
                (Object[]) filter.gradient.toDoubles()));

        System.out.println(deepToString(Conv2DData_3x3_2x2_Filter.grad_input));
        System.out.println(deepToString((Object[]) input.gradient.toDoubles()));
        assertTrue(deepEquals(Conv2DData_3x3_2x2_Filter.grad_input,
                (Object[]) input.gradient.toDoubles()));

    }

    @Test
    public void conv2D() {
        Tensor input = new Tensor(Conv2DData_1_Filter.input);
        Tensor filter = new Tensor(Conv2DData_1_Filter.filter);

        Tensor y = Ops.conv2d(input, filter);
        y.backward(new TArray(Conv2DData_1_Filter.grad_y));

        assertTrue(deepEquals(Conv2DData_1_Filter.y,
                (Object[]) y.vals.toDoubles()));

        System.out.println(deepToString(Conv2DData_1_Filter.grad_filter));
        System.out.println(deepToString((Object[]) filter.gradient.toDoubles()));
        assertTrue(deepEquals(Conv2DData_1_Filter.grad_filter,
                (Object[]) filter.gradient.toDoubles()));

        //System.out.println(deepToString(Conv2DData_1_Filter.grad_input));
        //System.out.println(deepToString((Object[]) input.gradient.m.toArray()));
        assertTrue(deepEquals(Conv2DData_1_Filter.grad_input,
                (Object[]) input.gradient.toDoubles()));

    }

    @Test
    public void conv2D2_1() {
        Tensor input = new Tensor(Conv2DData_2in_1out.input);
        Tensor filter = new Tensor(Conv2DData_2in_1out.filter);

        Tensor y = Ops.conv2d(input, filter);
        y.backward(new TArray(Conv2DData_2in_1out.grad_y));

        //System.out.println(deepToString(Conv2DData_2in_1out.y));
        //System.out.println(deepToString((Object[]) y.m.toArray()));
        assertTrue(deepEquals(Conv2DData_2in_1out.y,
                (Object[]) y.vals.toDoubles()));

        System.out.println(deepToString(Conv2DData_2in_1out.grad_input));
        System.out.println(deepToString((Object[]) filter.gradient.toDoubles()));
        assertTrue(deepEquals(Conv2DData_2in_1out.grad_filter,
                (Object[]) filter.gradient.toDoubles()));

        //System.out.println(deepToString(Conv2DData_2in_1out.grad_input));
        //System.out.println(deepToString((Object[]) input.gradient.m.toArray()));
        assertTrue(deepEquals(Conv2DData_2in_1out.grad_input,
                (Object[]) input.gradient.toDoubles()));
    }

    @Test
    public void conv2D2() {
        Tensor input = new Tensor(Conv2DData_2in_2out.input);
        Tensor filter = new Tensor(Conv2DData_2in_2out.filter);

        Tensor y = Ops.conv2d(input, filter);
        y.backward(new TArray(Conv2DData_2in_2out.grad_y));

        assertTrue(deepEquals(Conv2DData_2in_2out.y,
                (Object[]) y.vals.toDoubles()));

        //System.out.println(deepToString(Conv2DData_3x3_2x2_Filter.grad_filter));
        //System.out.println(deepToString((Object[]) filter.gradient.m.toArray()));
        assertTrue(deepEquals(Conv2DData_2in_2out.grad_filter,
                (Object[]) filter.gradient.toDoubles()));

        System.out.println(deepToString(Conv2DData_2in_2out.grad_input));
        System.out.println(deepToString((Object[]) input.gradient.toDoubles()));
        assertTrue(deepEquals(Conv2DData_2in_2out.grad_input,
                (Object[]) input.gradient.toDoubles()));
    }

    @Test
    public void conv2D_2x2_4x4() {
        Tensor input = new Tensor(Conv2DData_2x2_4x4.input);
        Tensor filter = new Tensor(Conv2DData_2x2_4x4.filter);

        Tensor y = Ops.conv2d(input, filter);
        y.backward(new TArray(Conv2DData_2x2_4x4.grad_y));

        assertTrue(deepEquals(Conv2DData_2x2_4x4.y,
                (Object[]) y.vals.toDoubles()));

        System.out.println(deepToString(Conv2DData_2x2_4x4.grad_filter));
        System.out.println(deepToString((Object[]) filter.gradient.toDoubles()));
        assertTrue(deepEquals(Conv2DData_2x2_4x4.grad_filter,
                (Object[]) filter.gradient.toDoubles()));

        System.out.println(deepToString(Conv2DData_2x2_4x4.grad_input));
        System.out.println(deepToString((Object[]) input.gradient.toDoubles()));
        assertTrue(deepEquals(Conv2DData_2x2_4x4.grad_input,
                (Object[]) input.gradient.toDoubles()));
    }
}
