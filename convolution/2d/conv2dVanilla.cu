#include <iostream>
#include <vector>
#include <cuda_runtime.h>


__constant__ float F[7 * 7];

void conv2dCpu(const float* A, float* out, const float* F, int width, int height, int R){
    for(int i=0; i<height; ++i){
        for(int j=0; j<width; ++j){
            float sum = 0.0;
            for(int m=-R; m<=R; ++m){
                for(int n=-R; n<=R; ++n){
                    int offset_i = i + m;
                    int offset_j = j + n;
                    if(
                        offset_i >= 0 && offset_i < height &&
                        offset_j >= 0 && offset_j < width
                    ){
                        sum += A[offset_i * width + offset_j] * \
                               F[(m + R) * (2 * R + 1) + (n + R)];
                    }
                }
            }
            out[i * width + j] = sum;
        }
    }
}

__global__ void kernConv2d(const float* A, float* out, int width, int height, int R){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if(row >= height || col >= width){
        return;
    }
    float sum = 0.0;
    for(int m=-R; m<=R; ++m){
        for(int n=-R; n<=R; ++n){
            int offset_i = row + m;
            int offset_j = col + n;
            if(
                offset_i >= 0 && offset_i < height &&
                offset_j >= 0 && offset_j < width
            ){
                sum += A[offset_i * width + offset_j] * \
                       F[(m + R) * (2 * R + 1) + (n + R)];
            }
        }
    }
    out[row * width + col] = sum;
}

int main() {
    // 1. Setup Large Dimensions
    const int width = 16384;
    const int height = 16384;
    const int R = 3; // 7x7 filter
    const int filterSide = 2 * R + 1;
    
    size_t imgSize = width * height * sizeof(float);
    size_t filterSize = filterSide * filterSide * sizeof(float);

    // 2. Allocate Host Memory
    std::vector<float> h_A(width * height, 1.0f);
    std::vector<float> h_F(filterSide * filterSide, 1.0f / (filterSide * filterSide)); // Box blur
    std::vector<float> h_out(width * height);

    // 3. Allocate Device Memory
    float *d_A, *d_F, *d_out;
    cudaMalloc(&d_A, imgSize);
    cudaMalloc(&d_F, filterSize);
    cudaMalloc(&d_out, imgSize);

    // 4. Copy to Device
    cudaMemcpy(d_A, h_A.data(), imgSize, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_F, h_F.data(), filterSize, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(F, h_F.data(), filterSize);

    // 5. Profiling Setup
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 6. Warm-up Run (GPU needs to wake up)
    kernConv2d<<<gridSize, blockSize>>>(d_A, d_out, width, height, R);

    // 7. Profiled Run
    cudaEventRecord(start);
    int iterations = 10;
    for(int i = 0; i < iterations; i++) {
        kernConv2d<<<gridSize, blockSize>>>(d_A, d_out, width, height, R);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Avg Execution Time: " << (milliseconds / iterations) << " ms" << std::endl;
    std::cout << "Bandwidth: " << (imgSize * 2.0 / (milliseconds / iterations / 1000.0) / 1e9) << " GB/s" << std::endl;

    // Cleanup
    cudaFree(d_A); cudaFree(d_F); cudaFree(d_out);
    return 0;
}