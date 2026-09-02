#include <iostream>
#include <cmath>
#include "../src/pool_s2.h"

int main() {
    static float input[IN_H][IN_W][CHANNELS];
    static float output[OUT_H][OUT_W][CHANNELS];

    // Simple test: fill input with known values so we can hand-check the result.
    // Use the same pattern as your conv tests: 0,1,2,3... row-major per channel.
    for (int r = 0; r < IN_H; r++)
        for (int c = 0; c < IN_W; c++)
            for (int ch = 0; ch < CHANNELS; ch++)
                input[r][c][ch] = (float)(r * IN_W + c);

    pool_s2(input, output);

    // Check the top-left output window by hand:
    // window = input[0][0], input[0][1], input[1][0], input[1][1]
    //        = 0, 1, 28, 29  -> average = 58/4 = 14.5
    float expected = 14.5f;
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