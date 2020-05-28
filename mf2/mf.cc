#include "mf.h"
#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;
void mf(int ny, int nx, int hy, int hx, const float* in, float* out) {
    
    #pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < ny; i++) {
        vector<float> pixels;
        for (int j = 0; j < nx; j++) {
            
            for (int k = max(0, i - hy); k <= min(ny-1, i + hy); k++) {
                for (int l = max(0, j - hx); l <= min(nx-1, j + hx); l++) {
                    pixels.push_back(in[l + k * nx]);
                }
            }
            
            int middle = pixels.size() / 2;
            nth_element(pixels.begin(), pixels.begin() + middle, pixels.end());
            
            if (pixels.size() % 2 == 0)
                out[j + i * nx] = (pixels[middle] + *max_element(pixels.begin(), pixels.begin() + middle)) / 2;
            else
                out[j + i * nx] = pixels[middle];
            
            pixels.erase(pixels.begin(), pixels.end());
        }
    }
    
}

