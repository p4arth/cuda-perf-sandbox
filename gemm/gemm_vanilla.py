import math
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def gemm_kernel(
    gA: cute.Tensor, 
    gB: cute.Tensor, 
    gC: cute.Tensor,
    M: cute.Int32, 
    N: cute.Int32, 
    K: cute.Int32,
    alpha: cute.Float32,
    beta: cute.Float32, 
):
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()
    row = bidx * bdimx + tidx
    col = bidy * bdimy + tidy
    # boundary-checks
    if (row < M) & (col < N):
        acc = 0.0
        for _k in cutlass.range(K):
            acc += gA[row * K + _k] * gB[N * _k + col]
        tmp = gC[row * N + col]
        gC[row * N + col] = alpha * acc + beta * tmp

@cute.jit
def gemm(
    gA:    cute.Tensor, 
    gB:    cute.Tensor, 
    gC:    cute.Tensor,
    M:     cute.Int32, 
    N:     cute.Int32, 
    K:     cute.Int32,
    alpha: cute.Float32,
    beta:  cute.Float32, 
):
    n_thrd_x = 32
    n_thrd_y = 32
    grid_x = (M + n_thrd_x - 1) // n_thrd_x
    grid_y = (N + n_thrd_y - 1) // n_thrd_y
    gemm_kernel(gA, gB, gC, M, N, K, alpha, beta).launch(
        grid=(grid_x, grid_y, 1),
        block=(n_thrd_x, n_thrd_y, 1)
    )

ALPHA = 1.0
BETA = 0.5
M, N, K = 1024, 1024, 1024
A = torch.randn((M, K), device="cuda")
B = torch.randn((K, N), device="cuda")
C = torch.zeros((M, N), device="cuda")
vadd_compiled = cute.compile(
    gemm, 
    from_dlpack(A), 
    from_dlpack(B), 
    from_dlpack(C), 
    M, N, K, ALPHA, BETA
)
vadd_compiled(
    from_dlpack(A), 
    from_dlpack(B), 
    from_dlpack(C), 
    M, N, K, ALPHA, BETA
)
print(C)
