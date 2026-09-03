#include <iostream>
#include <cmath>
#include "../src/pool_s4.h"

int main() {
    static float input[IN_H][IN_W][CHANNELS];
    static float output[OUT_H][OUT_W][CHANNELS];

    // Simple test: fill input with known values so we can hand-check the result.
    // Same pattern as pool_s2_tb.cpp: 0,1,2,3... row-major per channel.
    for (int r = 0; r < IN_H; r++)
        for (int c = 0; c < IN_W; c++)
            for (int ch = 0; ch < CHANNELS; ch++)
                input[r][c][ch] = (float)(r * IN_W + c);

    pool_s4(input, output);

    // Check the top-left output window by hand:
    // window = input[0][0], input[0][1], input[1][0], input[1][1]
    //        = 0, 1, 10, 11  -> max = 11  (IN_W=10 here, unlike S2's 28)
    float expected = 11.0f;
    float actual = output[0][0][0];

    std::cout << "Expected top-left value: " << expected << std::endl;
    std::cout << "Actual top-left value:   " << actual << std::endl;

    if (std::fabs(expected - actual) < 1e-5) {
        std::cout << "TEST PASSED" << std::endl;
        return 0;
    } else {
        std::cout << "TEST FAILED" << std::endl;
        return 1;
    }
}
