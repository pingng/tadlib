package com.codeberry.tadlib.tensor.conv2ddata;

public class Conv2DData_3x3_2x2_Filter {
    // Dims: 1, 3, 3, 1,
    public static final double[][][][] input = new double[][][][]{{{{0.},
            {1.},
            {2.}},

            {{3.},
                    {4.},
                    {5.}},

            {{6.},
                    {7.},
                    {8.}}}};
    // Dims: 2, 2, 1, 1,
    public static final double[][][][] filter = new double[][][][]{{{{0.}},
            {{1.}}},
            {{{2.}},
                    {{3.}}}};
    // Dims: 1, 3, 3, 1,
    public static final double[][][][] y = new double[][][][]{{{{19.},
            {25.},
            {10.}},

            {{37.},
                    {43.},
                    {16.}},

            {{7.},
                    {8.},
                    {0.}}}};
    // Dims: 1, 3, 3, 1,
    public static final double[][][][] grad_y = new double[][][][]{{{{0.},
            {1.},
            {2.}},

            {{3.},
                    {4.},
                    {5.}},

            {{6.},
                    {7.},
                    {8.}}}};
    // Dims: 1, 3, 3, 1,
    public static final double[][][][] grad_input = new double[][][][]{{{{0.},
            {0.},
            {1.}},

            {{0.},
                    {5.},
                    {11.}},

            {{6.},
                    {23.},
                    {29.}}}};
    // Dims: 2, 2, 1, 1,
    public static final double[][][][] grad_filter = new double[][][][]{{{{204.}},
            {{132.}}},
            {{{100.}},
                    {{58.}}}};
}
