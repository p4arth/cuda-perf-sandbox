#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

void checkCudaError(cudaError_t err, const char* msg){
    if (err != cudaSuccess){
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void kernVectorAdd(const float* A, const float* B, float* C, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N){
        C[i] = A[i] + B[i];
    }
}


void runVectorAddTest(int N){
    size_t size = N * sizeof(float);
    std::cout << "Testing Vector Addition with " << N << " elements..." << std::endl;

    float *h_A, *h_B, *h_C;
    checkCudaError(cudaMallocHost(&h_A, size), "[HOST] Malloc A");
    checkCudaError(cudaMallocHost(&h_B, size), "[HOST] Malloc B");
    checkCudaError(cudaMallocHost(&h_C, size), "[HOST] Malloc C");

    for(int i = 0; i < N; ++i){
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    checkCudaError(cudaMalloc(&d_A, size), "Malloc A");
    checkCudaError(cudaMalloc(&d_B, size), "Malloc B");
    checkCudaError(cudaMalloc(&d_C, size), "Malloc C");

    int num_streams = 8;
    cudaStream_t streams[num_streams];
    for(int i=0; i<num_streams; ++i) cudaStreamCreate(&streams[i]);

    int chunk_size = N / num_streams;
    int numThreadsChunk = 1024;
    int numBlocksChunk = 256 * 512 / 2;

    for(int i = 0; i < num_streams; ++i){
        // Copy data from i to i * chunk_size
        int offset = i * chunk_size;
        checkCudaError(
            cudaMemcpyAsync(
                d_A + offset, h_A + offset, chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[i]
            ), 
            "Memcpy H2D A"
        );
        checkCudaError(
            cudaMemcpyAsync(
                d_B + offset, h_B + offset, chunk_size * sizeof(float), cudaMemcpyHostToDevice, streams[i]
            ), 
            "Memcpy H2D B"
        );
        kernVectorAdd<<<
            numBlocksChunk, 
            numThreadsChunk,
            0, 
            streams[i]
        >>>(
            d_A + offset,
            d_B + offset,
            d_C + offset,
            chunk_size
        );
        checkCudaError(
            cudaMemcpyAsync(
                h_C + offset, d_C + offset, chunk_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]
            ), 
            "Memcpy D2H C"
        );
    }

    checkCudaError(cudaGetLastError(), "Kernel Launch");
    checkCudaError(cudaDeviceSynchronize(), "Sync");
    
    bool success = true;
    for(int i = 0; i < N; i++){
        if(std::fabs((h_A[i] + h_B[i]) - h_C[i]) > 1e-5){
            success = false;
            break;
        }
    }
    if (success) std::cout << "SUCCESS: Results match CPU reference." << std::endl;

    // Free alloc memory
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(){
    runVectorAddTest(1048576 * 512);
    return 0; 
}
