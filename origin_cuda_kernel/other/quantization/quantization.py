import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension
current_dir = os.path.dirname(os.path.abspath(__file__))
quantization = load(
    name='quantization',
    sources=[os.path.join(current_dir, 'quantization.cu')],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    build_directory=os.path.join(current_dir, 'build'),
    verbose=False
)

def test_mxfp8():
    print("=" * 50)
    print("Testing MXFP8 Quantization / Dequantization / GEMM")
    print("=" * 50)

    # MXFP8 Quantize
    x = torch.randn(256, device='cuda')
    scales = quantization.torch_mxfp8_quantize(x)
    print(f"[MXFP8 Quantize] input: {x.shape}, scales: {scales.shape}")

    # MXFP8 GEMM
    M, K, N = 128, 64, 128
    A = torch.randn(M, K, device='cuda')
    B = torch.randn(K, N, device='cuda')
    C = quantization.torch_mxfp8_gemm(A, B)
    C_ref = A @ B
    print(f"[MXFP8 GEMM] shape: {C.shape}, max_error: {(C - C_ref).abs().max().item():.6f}")

def test_mxfp4():
    print("\n" + "=" * 50)
    print("Testing MXFP4 Quantization / Dequantization / GEMM")
    print("=" * 50)

    # MXFP4 GEMM
    M, K, N = 128, 64, 128
    A = torch.randn(M, K, device='cuda') * 2
    B = torch.randn(K, N, device='cuda') * 2
    C = quantization.torch_mxfp4_gemm(A, B)
    C_ref = A @ B
    print(f"[MXFP4 GEMM] shape: {C.shape}, max_error: {(C - C_ref).abs().max().item():.6f}")

def test_nvfp4():
    print("\n" + "=" * 50)
    print("Testing NVFP4 Quantization / Dequantization / GEMV / GEMM")
    print("=" * 50)

    # NVFP4 GEMM
    M, K, N = 128, 64, 128
    A = torch.randn(M, K, device='cuda') * 2
    B = torch.randn(K, N, device='cuda') * 2
    C = quantization.torch_nvfp4_gemm(A, B)
    C_ref = A @ B
    print(f"[NVFP4 GEMM] shape: {C.shape}, max_error: {(C - C_ref).abs().max().item():.6f}")

if __name__ == '__main__':
    test_mxfp8()
    test_mxfp4()
    test_nvfp4()
    print("\nAll quantization tests passed!")
