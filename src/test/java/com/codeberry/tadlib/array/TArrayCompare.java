package com.codeberry.tadlib.array;

import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.java.JavaProvider;
import com.codeberry.tadlib.provider.opencl.OpenCLProvider;
import com.codeberry.tadlib.util.MatrixTestUtils;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static com.codeberry.tadlib.array.Comparison.*;
import static com.codeberry.tadlib.util.MatrixTestUtils.*;
import static org.junit.jupiter.api.Assertions.*;

public class TArrayCompare {

    @BeforeEach
    public void init() {
//        ProviderStore.setProvider(new JavaProvider());
        ProviderStore.setProvider(new OpenCLProvider());
    }

    @Test
    public void greaterSimple() {
        NDArray doubleArray = ProviderStore.array(new double[]{6.0, 0.0, 3});
        NDIntArray intArray = ProviderStore.array(new int[]{5, 0, 2});

        NDArray actual = doubleArray.compare(intArray, greaterThan(), 10, 0);

        NDArray expected = ProviderStore.array(new double[]{10, 0, 10});

        assertEqualsMatrix(expected.toDoubles(), actual.toDoubles());
    }

    @Test
    public void lessSimple() {
        NDArray doubleArray = ProviderStore.array(new double[]{-6.0, 0.0, 3});
        NDIntArray intArray = ProviderStore.array(new int[]{5, 0, 2});

        NDArray actual = doubleArray.compare(intArray, lessThan(), 10, 0);

        NDArray expected = ProviderStore.array(new double[]{10, 0, 0});

        assertEqualsMatrix(expected.toDoubles(), actual.toDoubles());
    }

    @Test
    public void greaterThanEqualsSimple() {
        NDArray doubleArray = ProviderStore.array(new double[]{6.0, 0.0, 1});
        NDIntArray intArray = ProviderStore.array(new int[]{5, 0, 2});

        NDArray actual = doubleArray.compare(intArray, greaterThanOrEquals(), 10, 0);

        NDArray expected = ProviderStore.array(new double[]{10, 10, 0});

        assertEqualsMatrix(expected.toDoubles(), actual.toDoubles());
    }

    @Test
    public void lessThanEqualsSimple() {
        NDArray doubleArray = ProviderStore.array(new double[]{6.0, 0.0, 1});
        NDIntArray intArray = ProviderStore.array(new int[]{5, 0, 2});

        NDArray actual = doubleArray.compare(intArray, lessThanOrEquals(), 10, 0);

        NDArray expected = ProviderStore.array(new double[]{0, 10, 10});

        assertEqualsMatrix(expected.toDoubles(), actual.toDoubles());
    }

    @Test
    public void equalsSimple_DoubleVsInt() {
        NDArray doubleArray = ProviderStore.array(new double[]{5.0, 1.0, 2});
        NDIntArray intArray = ProviderStore.array(new int[]{5, 1, -1});

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
        NDIntArray intArray = ProviderStore.array(new int[]{6, 1, -3});
        NDArray doubleArray = ProviderStore.array(new double[]{5, 1, -1});

        {
            NDArray actual = intArray.compare(doubleArray, greaterThan(), 1.234, 0.987);
            NDArray expected = ProviderStore.array(new double[]{1.234, 0.987, 0.987});

            assertEqualsMatrix(expected.toDoubles(), actual.toDoubles());
        }
        {
            NDArray actual = intArray.compare(doubleArray, lessThan(), 1.234, 0.987);
            NDArray expected = ProviderStore.array(new double[]{0.987, 0.987, 1.234});

            assertEqualsMatrix(expected.toDoubles(), actual.toDoubles());
        }
    }

    @Test
    public void greaterAndLessSimple_IntVsInt() {
        NDIntArray intArray = ProviderStore.array(new int[]{6, 1, -3});
        NDIntArray intOtherArray = ProviderStore.array(new int[]{5, 1, -1});

        {
            NDIntArray actual = intArray.compare(intOtherArray, greaterThan(), 123, 0);
            NDIntArray expected = ProviderStore.array(new int[]{123, 0, 0});

            assertArrayEquals((int[]) expected.toInts(), (int[]) actual.toInts());
        }
        {
            NDIntArray actual = intArray.compare(intOtherArray, lessThan(), 123, 0);
            NDIntArray expected = ProviderStore.array(new int[]{0, 0, 123});

            assertArrayEquals((int[]) expected.toInts(), (int[]) actual.toInts());
        }
    }

    @Test
    public void equalsScalar() {
        NDArray doubleArray = ProviderStore.array(5.0);
        NDIntArray intArray = ProviderStore.array(5);

        NDArray actual = doubleArray.compare(intArray, equalsWithDelta(1.0E-10), 1.234, 0.987);

        NDArray expected = ProviderStore.array(1.234);

        assertEquals((Double) expected.toDoubles(), (Double) actual.toDoubles());
    }

    @Test
    public void equalsBroadcast() {
        {
            NDArray doubleArray = ProviderStore.array(new double[]{5.0, 1.0, 2});
            NDIntArray intArray = ProviderStore.array(new int[]{5});

            NDArray actual = doubleArray.compare(intArray, equalsWithDelta(1.0E-10), 1.234, 0.987);
            NDArray expected = ProviderStore.array(new double[]{1.234, 0.987, 0.987});
            assertEqualsMatrix(expected.toDoubles(), actual.toDoubles());
        }
        {
            NDArray doubleArray = TArrayFactory.range(3 * 3).reshape(3, 3);
            NDIntArray intArray = ProviderStore.array(new int[]{0, 4, 8});

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
            NDIntArray intArray = TArrayFactory.intRange(3 * 3).reshape(3, 3);

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
