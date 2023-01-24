package com.codeberry.tadlib.tensor.conv2ddata;

public class Conv2DData_3x3_2x2_Filter {
    // Dims: 1, 3, 3, 1,
    public static final double[][][][] input = {
            {
                    {{0.0}, {10.0}, {2.0}},
                    {{3.0}, {4.0}, {50.0}},
                    {{60.0}, {7.0}, {8.0}}
            }
    };
    // Dims: 2, 2, 1, 1,
    public static final double[][][][] filter = {
            {{{0.0}}, {{1.0}}},
            {{{2.0}}, {{3.0}}}
    };
    // Dims: 1, 3, 3, 1,
    public static final double[][][][] y = {
            {
                    {{28.0}, {160.0}, {100.0}},
                    {{145.0}, {88.0}, {16.0}},
                    {{7.0}, {8.0}, {0.0}}
            }
    };
    // Dims: 1, 3, 3, 1,
    public static final double[][][][] grad_y = {
            {
                    {{0.0}, {-10000.0}, {2.0}},
                    {{300.0}, {4.0}, {5.0}},
                    {{6.0}, {7.0}, {8000.0}}
            }
    };
    // Dims: 1, 3, 3, 1,
    public static final double[][][][] grad_input = {
            {
                    {{0.0}, {0.0}, {-10000}},
                    {{0.0}, {-19700}, {-29992}},
                    {{600}, {914}, {29.0}}
            }
    };
    // Dims: 2, 2, 1, 1,
    public static final double[][][][] grad_filter = {
            {{{-34421}}, {{-18502}}},
            {{{-21832}}, {{-497868}}}
    };
}
