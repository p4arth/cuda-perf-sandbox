#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <cuda_runtime.h>

// CPU kernel for verification
void conv1dCpu(const float* A, float* out, const float* F, int N, int R){
    for(int i=0; i<N; ++i){
        float sum = 0.0;
        for(int j=0; j<(2*R + 1); j++){
            int offset = i + j - R;
            if(offset >= 0 && offset < N){
                sum +=  A[offset] * F[j];
            }
        }
        out[i] = sum;
    }
}

// GPU Kernel
__global__ void kernConv1d(const float* A, float* out, const float* F, int N, int R){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N){
        float sum = 0;
        for(int j=0; j<(2*R+1); j++){
            int offset = i + j - R;
            if(offset >= 0 && offset < N){
                sum += A[offset] * F[j];
            }
        }
        out[i] = sum;
    }
}
