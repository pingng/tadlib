package com.codeberry.tadlib.tensor.conv2ddata;

public class Conv2DData_3x3_2x2_Filter {
    // Dims: 1, 3, 3, 1,
    public static final double[][][][] input = new double[][][][]{
            {
                    {{0.}, {10.}, {2.}},
                    {{3.}, {4.}, {50.}},
                    {{60.}, {7.}, {8.}}
            }
    };
    // Dims: 2, 2, 1, 1,
    public static final double[][][][] filter = new double[][][][]{
            {{{0.}}, {{1.}}},
            {{{2.}}, {{3.}}}
    };
    // Dims: 1, 3, 3, 1,
    public static final double[][][][] y = new double[][][][]{
            {
                    {{28.}, {160.}, {100.}},
                    {{145.}, {88.}, {16.}},
                    {{7.}, {8.}, {0.}}
            }
    };
    // Dims: 1, 3, 3, 1,
    public static final double[][][][] grad_y = new double[][][][]{
            {
                    {{0.}, {-10000.}, {2.}},
                    {{300.}, {4.}, {5.}},
                    {{6.}, {7.}, {8000.}}
            }
    };
    // Dims: 1, 3, 3, 1,
    public static final double[][][][] grad_input = new double[][][][]{
            {
                    {{0.}, {0.}, {-10000}},
                    {{0.}, {-19700}, {-29992}},
                    {{600}, {914}, {29.}}
            }
    };
    // Dims: 2, 2, 1, 1,
    public static final double[][][][] grad_filter = new double[][][][]{
            {{{-34421}}, {{-18502}}},
            {{{-21832}}, {{-497868}}}
    };
}
