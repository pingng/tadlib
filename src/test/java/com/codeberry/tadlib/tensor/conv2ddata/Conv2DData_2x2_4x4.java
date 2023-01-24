package com.codeberry.tadlib.tensor.conv2ddata;

public class Conv2DData_2x2_4x4 {
    // Dims: 1, 2, 2, 1,
    public static final double[][][][] input = {{{{0.0},
            {1.0}},

            {{2.0},
                    {3.0}}}};
    // Dims: 4, 4, 1, 1,
    public static final double[][][][] filter = {{{{0.0}},

            {{1.0}},

            {{2.0}},

            {{3.0}}},


            {{{4.0}},

                    {{5.0}},

                    {{6.0}},

                    {{7.0}}},


            {{{8.0}},

                    {{9.0}},

                    {{10.0}},

                    {{11.0}}},


            {{{12.0}},

                    {{13.0}},

                    {{14.0}},

                    {{15.0}}}};
    // Dims: 1, 2, 2, 1,
    public static final double[][][][] y = {{{{54.0},
            {48.0}},

            {{30.0},
                    {24.0}}}};
    // Dims: 1, 2, 2, 1,
    public static final double[][][][] grad_y = {{{{0.0},
            {1.0}},

            {{2.0},
                    {3.0}}}};
    // Dims: 1, 2, 2, 1,
    public static final double[][][][] grad_input = {{{{6.0},
            {12.0}},

            {{30.0},
                    {36.0}}}};
    // Dims: 4, 4, 1, 1,
    public static final double[][][][] grad_filter = {{{{0.0}},

            {{3.0}},

            {{2.0}},

            {{0.0}}},


            {{{6.0}},

                    {{14.0}},

                    {{6.0}},

                    {{0.0}}},


            {{{2.0}},

                    {{3.0}},

                    {{0.0}},

                    {{0.0}}},


            {{{0.0}},

                    {{0.0}},

                    {{0.0}},

                    {{0.0}}}};
}
