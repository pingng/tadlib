package com.codeberry.tadlib.array;

import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.java.JavaIntArray;
import com.codeberry.tadlib.provider.java.NDArray;
import org.junit.jupiter.api.Test;

import static com.codeberry.tadlib.array.Comparison.*;
import static com.codeberry.tadlib.util.MatrixTestUtils.assertEqualsMatrix;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class TArrayCompare {

    @Test
    public void greaterSimple() {
        NDArray doubleArray = ProviderStore.array(new double[]{6.0, 0.0, 3});
        JavaIntArray intArray = ProviderStore.array(new int[]{5, 0, 2});

        NDArray actual = doubleArray.compare(intArray, greaterThan, 10, 0);

        NDArray expected = ProviderStore.array(new double[]{10, 0, 10});

        assertEqualsMatrix(expected.toDoubles(), actual.toDoubles());
    }

    @Test
    public void lessSimple() {
        NDArray doubleArray = ProviderStore.array(new double[]{-6.0, 0.0, 3});
        JavaIntArray intArray = ProviderStore.array(new int[]{5, 0, 2});

        NDArray actual = doubleArray.compare(intArray, lessThan, 10, 0);

        NDArray expected = ProviderStore.array(new double[]{10, 0, 0});

        assertEqualsMatrix(expected.toDoubles(), actual.toDoubles());
    }

    @Test
    public void greaterThanEqualsSimple() {
        NDArray doubleArray = ProviderStore.array(new double[]{6.0, 0.0, 1});
        JavaIntArray intArray = ProviderStore.array(new int[]{5, 0, 2});

        NDArray actual = doubleArray.compare(intArray, greaterThanOrEquals, 10, 0);

        NDArray expected = ProviderStore.array(new double[]{10, 10, 0});

        assertEqualsMatrix(expected.toDoubles(), actual.toDoubles());
    }

    @Test
    public void lessThanEqualsSimple() {
        NDArray doubleArray = ProviderStore.array(new double[]{6.0, 0.0, 1});
        JavaIntArray intArray = ProviderStore.array(new int[]{5, 0, 2});

        NDArray actual = doubleArray.compare(intArray, lessThanOrEquals, 10, 0);

        NDArray expected = ProviderStore.array(new double[]{0, 10, 10});

        assertEqualsMatrix(expected.toDoubles(), actual.toDoubles());
    }

    @Test
    public void equalsSimple_DoubleVsInt() {
        NDArray doubleArray = ProviderStore.array(new double[]{5.0, 1.0, 2});
        JavaIntArray intArray = ProviderStore.array(new int[]{5, 1, -1});

        NDArray actual = doubleArray.compare(intArray, equalsWithDelta(1.0E-10), 1.234, 0.987);

        NDArray expected = ProviderStore.array(new double[]{1.234, 1.234, 0.987});

        assertEqualsMatrix(expected.toDoubles(), actual.toDoubles());
    }

    @Test
    public void equalsSimple_DoubleVsDouble() {
        NDArray doubleArray = ProviderStore.array(new double[]{5.0, 1.0, 2});
        NDArray doubleOtherArray = ProviderStore.array(new double[]{5, 1, -1});

        NDArray actual = doubleArray.compare(doubleOtherArray, equalsWithDelta(1.0E-10), 1.234, 0.987);

        NDArray expected = ProviderStore.array(new double[]{1.234, 1.234, 0.987});

        assertEqualsMatrix(expected.toDoubles(), actual.toDoubles());
    }

    @Test
    public void greaterAndLessSimple_IntVsDouble() {
        JavaIntArray intArray = ProviderStore.array(new int[]{6, 1, -3});
        NDArray doubleArray = ProviderStore.array(new double[]{5, 1, -1});

        {
            NDArray actual = intArray.compare(doubleArray, greaterThan, 1.234, 0.987);
            NDArray expected = ProviderStore.array(new double[]{1.234, 0.987, 0.987});

            assertEqualsMatrix(expected.toDoubles(), actual.toDoubles());
        }
        {
            NDArray actual = intArray.compare(doubleArray, lessThan, 1.234, 0.987);
            NDArray expected = ProviderStore.array(new double[]{0.987, 0.987, 1.234});

            assertEqualsMatrix(expected.toDoubles(), actual.toDoubles());
        }
    }

    @Test
    public void greaterAndLessSimple_IntVsInt() {
        JavaIntArray intArray = ProviderStore.array(new int[]{6, 1, -3});
        JavaIntArray intOtherArray = ProviderStore.array(new int[]{5, 1, -1});

        {
            JavaIntArray actual = intArray.compare(intOtherArray, greaterThan, 123, 0);
            JavaIntArray expected = ProviderStore.array(new int[]{123, 0, 0});

            assertArrayEquals((int[]) expected.toInts(), (int[]) actual.toInts());
        }
        {
            JavaIntArray actual = intArray.compare(intOtherArray, lessThan, 123, 0);
            JavaIntArray expected = ProviderStore.array(new int[]{0, 0, 123});

            assertArrayEquals((int[]) expected.toInts(), (int[]) actual.toInts());
        }
    }

    @Test
    public void equalsScalar() {
        NDArray doubleArray = ProviderStore.array(5.0);
        JavaIntArray intArray = ProviderStore.array(5);

        NDArray actual = doubleArray.compare(intArray, equalsWithDelta(1.0E-10), 1.234, 0.987);

        NDArray expected = ProviderStore.array(1.234);

        assertEquals((Double) expected.toDoubles(), (Double) actual.toDoubles());
    }

    @Test
    public void equalsBroadcast() {
        {
            NDArray doubleArray = ProviderStore.array(new double[]{5.0, 1.0, 2});
            JavaIntArray intArray = ProviderStore.array(new int[]{5});

            NDArray actual = doubleArray.compare(intArray, equalsWithDelta(1.0E-10), 1.234, 0.987);
            NDArray expected = ProviderStore.array(new double[]{1.234, 0.987, 0.987});
            assertEqualsMatrix(expected.toDoubles(), actual.toDoubles());
        }
        {
            NDArray doubleArray = TArrayFactory.range(3 * 3).reshape(3, 3);
            JavaIntArray intArray = ProviderStore.array(new int[]{0, 4, 8});

            NDArray actual = doubleArray.compare(intArray, equalsWithDelta(1.0E-10), 123.5, 0);
            NDArray expected = ProviderStore.array(new double[]{
                    123.5, 0, 0,
                    0, 123.5, 0,
                    0, 0, 123.5
            }).reshape(3, 3);
            assertEqualsMatrix(expected.toDoubles(), actual.toDoubles());
        }
        {
            NDArray doubleArray = ProviderStore.array(new double[]{4});
            JavaIntArray intArray = TArrayFactory.intRange(3 * 3).reshape(3, 3);

            NDArray actual = doubleArray.compare(intArray, equalsWithDelta(1.0E-10), 0, -1);
            NDArray expected = ProviderStore.array(new double[]{
                    -1, -1, -1,
                    -1, 0, -1,
                    -1, -1, -1
            }).reshape(3, 3);
            assertEqualsMatrix(expected.toDoubles(), actual.toDoubles());
        }
    }
}
