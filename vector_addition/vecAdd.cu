#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

__global__ void kernVectorAdd(const float* A, const float* B, float* C, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N){
        C[i] = A[i] + B[i];
    }
}

void checkCudaError(cudaError_t err, const char* msg){
    if (err != cudaSuccess){
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void runVectorAddTest(int N){
    size_t size = N * sizeof(float);
    std::cout << "Testing Vector Addition with " << N << " elements..." << std::endl;

    std::vector<float> h_A(N), h_B(N), h_C(N);
    for(int i = 0; i < N; ++i){
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    checkCudaError(cudaMalloc(&d_A, size), "Malloc A");
    checkCudaError(cudaMalloc(&d_B, size), "Malloc B");
    checkCudaError(cudaMalloc(&d_C, size), "Malloc C");

    // Copy data from host to device
    checkCudaError(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice), "Memcpy H2D A");
    checkCudaError(cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice), "Memcpy H2D B");

    int numThreads = 256;
    int numBlocks = (N + numThreads-1)/numThreads;

    kernVectorAdd<<<numBlocks, numThreads>>>(
        d_A,
        d_B,
        d_C,
        N
    );

    checkCudaError(cudaGetLastError(), "Kernel Launch");
    checkCudaError(cudaDeviceSynchronize(), "Sync");

    // Copy results back to host
    checkCudaError(cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost), "Memcpy D2H C");

    bool success = true;
    for(int i = 0; i < N; i++){
        if(std::fabs(h_A[i]+ h_B[i] - h_C[i] > 1e-5)){
            std::cerr << "Verification failed at element " << i << std::endl;
            success = false;
            break;
        }
    }
    if (success) std::cout << "SUCCESS: Results match CPU reference." << std::endl;

    // Free alloc memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(){
    runVectorAddTest(1048576);
    return 0; 
}
