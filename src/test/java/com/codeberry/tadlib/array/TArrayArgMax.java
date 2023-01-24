package com.codeberry.tadlib.array;

import com.codeberry.tadlib.array.exception.AxisOutOfBounds;
import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.java.JavaIntArray;
import com.codeberry.tadlib.provider.java.NDArray;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

public class TArrayArgMax {

    @Test
    public void axisOutOfBounds() {
        NDArray input = ProviderStore.array(new double[]{1, 2, 3, 0});

        assertThrows(AxisOutOfBounds.class, () ->
                input.argmax(1));
        assertThrows(AxisOutOfBounds.class, () ->
                input.argmax(-2));
    }

    @Test
    public void testVectorMax() {
        //@formatter:off
        NDArray input = ProviderStore.array(new double[]{
                1, 2, 3, 0
        });
        //@formatter:on

        JavaIntArray indices = input.argmax(0);

        Integer maxIndex = (Integer) indices.toInts();
        assertEquals(2, maxIndex);
    }

    @Test
    public void test3DimsAtFirstAxis() {
        //@formatter:off
        NDArray input = ProviderStore.array(new double[]{
                1, 2, 3, 0, 3, 100,
                6, 2, 1, -5, 4, 3,
                5, 20, 2, 1, -5, 20
        }).reshape(3, 2, 3);
        //@formatter:on

        JavaIntArray indices = input.argmax(0);

        int[][] ints = (int[][]) indices.toInts();
        System.out.println(Arrays.deepToString(ints));
        assertTrue(Arrays.deepEquals(new int[][]{
                {1, 2, 0},
                {2, 1, 0},
        }, ints));
    }

    @Test
    public void test3DimsAtSecondAxis() {
        //@formatter:off
        NDArray input = ProviderStore.array(new double[]{
                1, 2, 3, 0, 3, 100,
                6, 2, 1, -5, 4, 3,
                5, 20, 2, 1, -5, 20
        }).reshape(3, 2, 3);
        //@formatter:on

        JavaIntArray indices = input.argmax(1);

        int[][] ints = (int[][]) indices.toInts();
        System.out.println(Arrays.deepToString(ints));
        assertTrue(Arrays.deepEquals(new int[][]{
                {0, 1, 1},
                {0, 1, 1},
                {0, 0, 1},
        }, ints));
    }

    @Test
    public void test3DimsAtLastAxis() {
        //@formatter:off
        NDArray input = ProviderStore.array(new double[]{
                1, 2, 3, 0, 3, 100,
                6, 2, 1, -5, 4, 3,
                5, 20, 2, 1, -5, 20
        }).reshape(3, 2, 3);
        //@formatter:on

        JavaIntArray indices = input.argmax(2);

        int[][] ints = (int[][]) indices.toInts();
        System.out.println(Arrays.deepToString(ints));
        assertTrue(Arrays.deepEquals(new int[][]{
                {2, 2},
                {0, 1},
                {1, 2},
        }, ints));
    }

    @Test
    public void test2DimsAtFirstAxis() {
        //@formatter:off
        NDArray input = ProviderStore.array(new double[]{
                1, 2, 3,
                0, 3, 100,
                6, 2, 1,
                -5, 4, 3,
                5, 20, 2,
                1, -5, 20
        }).reshape(6, 3);
        //@formatter:on

        JavaIntArray indices = input.argmax(0);

        int[] ints = (int[]) indices.toInts();
        System.out.println(Arrays.toString(ints));
        assertArrayEquals(new int[]{
                2, 4, 1
        }, ints);
    }

    @Test
    public void test2DimsAtLastAxis() {
        //@formatter:off
        NDArray input = ProviderStore.array(new double[]{
                1, 2, 3,
                0, 3, 100,
                6, 2, 1,
                -5, 4, 3,
                5, 20, 2,
                1, -5, 20
        }).reshape(6, 3);
        //@formatter:on

        JavaIntArray indices = input.argmax(-1);

        int[] ints = (int[]) indices.toInts();
        System.out.println(Arrays.toString(ints));
        assertArrayEquals(new int[]{
                2,
                2,
                0,
                1,
                1,
                2
        }, ints);
    }

}
