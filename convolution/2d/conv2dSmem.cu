#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define RADIUS 3
#define B_DIM 32
#define F_SIDE (2 * RADIUS + 1)
__constant__ float F[F_SIDE * F_SIDE];

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

__global__ void kernConv2d(const float* A, float* out, int width, int height){
    __shared__ float tile[B_DIM*B_DIM];
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * (blockDim.x - 2*RADIUS) + tx - RADIUS;
    int row = blockIdx.y * (blockDim.y - 2*RADIUS) + ty - RADIUS;
    if(row >= 0 && row < height && col >= 0 && col < width){
        tile[ty * blockDim.x + tx] = A[row * width + col];
    }
    else{
        tile[ty * blockDim.x + tx] = 0.0;
    }
    __syncthreads();

    if(tx >= RADIUS && tx < blockDim.x - RADIUS && ty >= RADIUS && ty < blockDim.y - RADIUS){
        if (row >= 0 && row < height && col >= 0 && col < width){
            float sum = 0.0f;
            #pragma unroll
            for (int m = -RADIUS; m <= RADIUS; ++m) {
                #pragma unroll
                for (int n = -RADIUS; n <= RADIUS; ++n) {
                    float pixel = tile[(ty + m) * B_DIM + (tx + n)];
                    float coeff = F[(m + RADIUS) * (2 * RADIUS + 1) + (n + RADIUS)];
                    sum += pixel * coeff;
                }
            }
            out[row * width + col] = sum;
        }
    }
}

int main() {
    // 1. Setup Large Dimensions
    const int width = 4096;
    const int height = 4096;
    const int filterSide = 2 * RADIUS + 1;
    
    size_t imgSize = width * height * sizeof(float);
    size_t filterSize = filterSide * filterSide * sizeof(float);

    float *h_A, *h_F, *h_out;
    cudaMallocHost((void**)&h_A, imgSize);
    cudaMallocHost((void**)&h_F, filterSize);
    cudaMallocHost((void**)&h_out, imgSize);

    for (int i = 0; i < width * height; i++) {
        h_A[i] = 1.0f;
    }

    // 3. Allocate Device Memory
    float *d_A, *d_F, *d_out;
    cudaMalloc(&d_A, imgSize);
    cudaMalloc(&d_F, filterSize);
    cudaMalloc(&d_out, imgSize);

    // 4. Copy to Device
    cudaMemcpy(d_A, h_A, imgSize, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(F, h_F, filterSize);

    // 5. Profiling Setup
    dim3 blockSize(32, 32);
    int gsx = blockSize.x - 2*RADIUS;
    int gsy = blockSize.y - 2*RADIUS;
    dim3 gridSize((width + gsx - 1) / gsx, 
                  (height + gsy - 1) / gsy);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 6. Warm-up Run (GPU needs to wake up)
    kernConv2d<<<gridSize, blockSize>>>(d_A, d_out, width, height);

    // 7. Profiled Run
    cudaEventRecord(start);
    // int iterations = 10;
    // for(int i = 0; i < iterations; i++) {
    kernConv2d<<<gridSize, blockSize>>>(d_A, d_out, width, height);
    // }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Avg Execution Time: " << (milliseconds) << " ms" << std::endl;
    std::cout << "Bandwidth: " << (imgSize * 2.0 / (milliseconds / 1000.0) / 1e9) << " GB/s" << std::endl;

    // Cleanup
    cudaFree(d_A); cudaFree(d_F); cudaFree(d_out);
    cudaFree(h_A); cudaFree(h_F); cudaFree(h_out);
    return 0;
}