package com.codeberry.tadlib.tensor.conv2ddata;

public class Conv2DData_2x2_4x4 {
    // Dims: 1, 2, 2, 1,
    public static final double[][][][] input = new double[][][][] {{{{0.},
            {1.}},

            {{2.},
                    {3.}}}};
    // Dims: 4, 4, 1, 1,
    public static final double[][][][] filter = new double[][][][] {{{{ 0.}},

            {{ 1.}},

            {{ 2.}},

            {{ 3.}}},


            {{{ 4.}},

                    {{ 5.}},

                    {{ 6.}},

                    {{ 7.}}},


            {{{ 8.}},

                    {{ 9.}},

                    {{10.}},

                    {{11.}}},


            {{{12.}},

                    {{13.}},

                    {{14.}},

                    {{15.}}}};
    // Dims: 1, 2, 2, 1,
    public static final double[][][][] y = new double[][][][] {{{{54.},
            {48.}},

            {{30.},
                    {24.}}}};
    // Dims: 1, 2, 2, 1,
    public static final double[][][][] grad_y = new double[][][][] {{{{0.},
            {1.}},

            {{2.},
                    {3.}}}};
    // Dims: 1, 2, 2, 1,
    public static final double[][][][] grad_input = new double[][][][] {{{{ 6.},
            {12.}},

            {{30.},
                    {36.}}}};
    // Dims: 4, 4, 1, 1,
    public static final double[][][][] grad_filter = new double[][][][] {{{{ 0.}},

            {{ 3.}},

            {{ 2.}},

            {{ 0.}}},


            {{{ 6.}},

                    {{14.}},

                    {{ 6.}},

                    {{ 0.}}},


            {{{ 2.}},

                    {{ 3.}},

                    {{ 0.}},

                    {{ 0.}}},


            {{{ 0.}},

                    {{ 0.}},

                    {{ 0.}},

                    {{ 0.}}}};
}
