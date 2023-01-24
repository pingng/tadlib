package com.codeberry.tadlib.tensor.conv2ddata;

public class Conv2DData_2in_1out {
    // Dims: 1, 5, 5, 2,
    public static final double[][][][] input = {{{{0.0, 1.0},
            {2.0, 3.0},
            {4.0, 5.0},
            {6.0, 7.0},
            {8.0, 9.0}},

            {{10.0, 11.0},
                    {12.0, 13.0},
                    {14.0, 15.0},
                    {16.0, 17.0},
                    {18.0, 19.0}},

            {{20.0, 21.0},
                    {22.0, 23.0},
                    {24.0, 25.0},
                    {26.0, 27.0},
                    {28.0, 29.0}},

            {{30.0, 31.0},
                    {32.0, 33.0},
                    {34.0, 35.0},
                    {36.0, 37.0},
                    {38.0, 39.0}},

            {{40.0, 41.0},
                    {42.0, 43.0},
                    {44.0, 45.0},
                    {46.0, 47.0},
                    {48.0, 49.0}}}};
    // Dims: 3, 3, 2, 1,
    public static final double[][][][] filter = {{{{0.0},
            {1.0}},

            {{2.0},
                    {3.0}},

            {{4.0},
                    {5.0}}},


            {{{6.0},
                    {7.0}},

                    {{8.0},
                            {9.0}},

                    {{10.0},
                            {11.0}}},


            {{{12.0},
                    {13.0}},

                    {{14.0},
                            {15.0}},

                    {{16.0},
                            {17.0}}}};
    // Dims: 1, 5, 5, 1,
    public static final double[][][][] y = {{{{780.0},
            {1250.0},
            {1526.0},
            {1802.0},
            {1180.0}},

            {{1806.0},
                    {2685.0},
                    {2991.0},
                    {3297.0},
                    {2070.0}},

            {{2946.0},
                    {4215.0},
                    {4521.0},
                    {4827.0},
                    {2970.0}},

            {{4086.0},
                    {5745.0},
                    {6051.0},
                    {6357.0},
                    {3870.0}},

            {{2028.0},
                    {2690.0},
                    {2822.0},
                    {2954.0},
                    {1660.0}}}};
    // Dims: 1, 5, 5, 1,
    public static final double[][][][] grad_y = {{{{0.0},
            {1.0},
            {2.0},
            {3.0},
            {4.0}},

            {{5.0},
                    {6.0},
                    {7.0},
                    {8.0},
                    {9.0}},

            {{10.0},
                    {11.0},
                    {12.0},
                    {13.0},
                    {14.0}},

            {{15.0},
                    {16.0},
                    {17.0},
                    {18.0},
                    {19.0}},

            {{20.0},
                    {21.0},
                    {22.0},
                    {23.0},
                    {24.0}}}};
    // Dims: 1, 5, 5, 2,
    public static final double[][][][] grad_input = {{{{16.0, 28.0},
            {52.0, 73.0},
            {82.0, 109.0},
            {112.0, 145.0},
            {112.0, 136.0}},

            {{108.0, 141.0},
                    {240.0, 294.0},
                    {312.0, 375.0},
                    {384.0, 456.0},
                    {336.0, 387.0}},

            {{318.0, 381.0},
                    {600.0, 699.0},
                    {672.0, 780.0},
                    {744.0, 861.0},
                    {606.0, 687.0}},

            {{528.0, 621.0},
                    {960.0, 1104.0},
                    {1032.0, 1185.0},
                    {1104.0, 1266.0},
                    {876.0, 987.0}},

            {{688.0, 760.0},
                    {1168.0, 1279.0},
                    {1234.0, 1351.0},
                    {1300.0, 1423.0},
                    {976.0, 1060.0}}}};
    // Dims: 3, 3, 2, 1,
    public static final double[][][][] grad_filter = {{{{5360.0},
            {5600.0}},

            {{6840.0},
                    {7130.0}},

            {{5520.0},
                    {5744.0}}},


            {{{7800.0},
                    {8050.0}},

                    {{9800.0},
                            {10100.0}},

                    {{7800.0},
                            {8030.0}}},


            {{{5520.0},
                    {5680.0}},

                    {{6840.0},
                            {7030.0}},

                    {{5360.0},
                            {5504.0}}}};
}
