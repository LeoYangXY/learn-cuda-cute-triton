"""轻量 NCU 测试脚本：只跑一次 flash_attention kernel，避免 benchmark 开销。"""
import torch
from exercise_cuda.triton.flash_attention.flash_attention_v1 import flash_attention

torch.manual_seed(0)
BATCH, N_HEADS, HEAD_DIM, N_CTX = 4, 32, 64, 4096
dtype = torch.float16
q = torch.randn((BATCH, N_HEADS, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").contiguous()
k = torch.randn((BATCH, N_HEADS, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").contiguous()
v = torch.randn((BATCH, N_HEADS, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").contiguous()
sm_scale = 1.0 / (HEAD_DIM ** 0.5)

# warmup
flash_attention(q, k, v, sm_scale)
torch.cuda.synchronize()
# profiled launch
flash_attention(q, k, v, sm_scale)
torch.cuda.synchronize()
print("done")
