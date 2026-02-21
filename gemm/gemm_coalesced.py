import math
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def gemm_kernel(
    gA:    cute.Tensor, 
    gB:    cute.Tensor, 
    gC:    cute.Tensor,
    M:     cute.Int32, 
    N:     cute.Int32, 
    K:     cute.Int32,
    alpha: cute.Float32,
    beta:  cute.Float32, 
):
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()
    col = bidx * bdimx + tidx
    row = bidy * bdimy + tidy
    # boundary-checks
    if (row < M) & (col < N):
        acc = 0.0
        for _k in cutlass.range(K):
            acc += gA[row * K + _k] * gB[N * _k + col]
        gC[row * N + col] = alpha * acc + beta * gC[row * N + col]

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
    grid_x = (N + n_thrd_y - 1) // n_thrd_y
    grid_y = (M + n_thrd_x - 1) // n_thrd_x
    gemm_kernel(gA, gB, gC, M, N, K, alpha, beta).launch(
        grid=(grid_x, grid_y, 1),
        block=(n_thrd_x, n_thrd_y, 1)
    )

ALPHA = 1.0
BETA = 0.5
M, N, K = 1024, 1024, 1024
A = torch.ones((M, K), device="cuda")
B = torch.ones((K, N), device="cuda")
C = torch.zeros((M, N), device="cuda")
gemm_compiled = cute.compile(
    gemm, 
    from_dlpack(A), 
    from_dlpack(B), 
    from_dlpack(C), 
    M, N, K, ALPHA, BETA
)

# import torch.utils.benchmark as benchmark

# for _ in range(10):
#     gemm_compiled(from_dlpack(A), from_dlpack(B), from_dlpack(C), M, N, K, ALPHA, BETA)
# for _ in range(10):
#     res_pt = torch.addmm(C, A, B, alpha=ALPHA, beta=BETA)

torch.cuda.synchronize()
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

NLAPS = 1
start_event.record()
for _ in range(NLAPS): # Run multiple times for stability
    gemm_compiled(from_dlpack(A), from_dlpack(B), from_dlpack(C), M, N, K, ALPHA, BETA)
end_event.record()
torch.cuda.synchronize()
cute_time = start_event.elapsed_time(end_event) / NLAPS

# --- 3. Profile PyTorch ---
start_event.record()
for _ in range(NLAPS):
    res_pt = torch.addmm(C, A, B, alpha=ALPHA, beta=BETA)
end_event.record()
torch.cuda.synchronize()
pt_time = start_event.elapsed_time(end_event) / NLAPS

print(f"CuTe Naive Kernel: {cute_time:.4f} ms")
print(f"PyTorch :  {pt_time:.4f} ms")
print(f"Speedup:   {cute_time / pt_time:.2f}x (PyTorch is faster)")

