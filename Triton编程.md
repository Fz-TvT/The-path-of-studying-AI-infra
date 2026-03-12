# RMSNorm算子
``` Python
import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(
    x_ptr, y_ptr, w_ptr,
    stride_x_row, stride_y_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr
):
    # 1. 映射 Program ID 到具体的行 (Row)
    row_idx = tl.program_id(0)
    x_ptr_row = x_ptr + row_idx * stride_x_row
    y_ptr_row = y_ptr + row_idx * stride_y_row

    # 2. 生成列的内存偏移量与掩码 (Mask)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # 3. 将当前行的数据和权重加载到 SRAM (转为 float32 防止累加时精度溢出)
    x = tl.load(x_ptr_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # 4. 计算平方和与 rsqrt (倒数均方根)
    x_sq = x * x
    sum_sq = tl.sum(x_sq, axis=0)
    rms = tl.math.rsqrt(sum_sq / N + eps)

    # 5. 归一化并乘以可学习的权重
    x_norm = x * rms
    out = x_norm * w

    # 6. 将结果写回 HBM
    tl.store(y_ptr_row + cols, out.to(x_ptr.dtype.element_ty), mask=mask)

def triton_rmsnorm(x, weight, eps=1e-6):
    # x shape: [batch_size * seq_len, hidden_dim]
    x_2d = x.view(-1, x.shape[-1])
    M, N = x_2d.shape
    y_2d = torch.empty_like(x_2d)

    # BLOCK_SIZE 取大于等于 hidden_dim 的最小 2 的幂次
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    # 启动 1D Grid，每个 program 负责一个 Token (一行)
    grid = (M,)
    rmsnorm_kernel[grid](
        x_2d, y_2d, weight,
        x_2d.stride(0), y_2d.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y_2d.view_as(x)
```


# RoPE算子

```Python
@triton.jit
def rope_inplace_kernel(
    q_ptr, k_ptr, cos_ptr, sin_ptr, pos_ptr,
    stride_q_tok, stride_q_head,
    stride_k_tok, stride_k_head,
    stride_cos_row,
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr # BLOCK_SIZE = head_dim // 2
):
    # 1. 确定当前处理的 Token 和 Head
    pid_tok = tl.program_id(0)
    pid_head = tl.program_id(1)

    # 2. 读取当前 Token 对应的绝对位置索引
    pos = tl.load(pos_ptr + pid_tok)

    # 3. 计算 Q 和 K 当前 Head 的基础内存偏移
    q_offset = pid_tok * stride_q_tok + pid_head * stride_q_head
    k_offset = pid_tok * stride_k_tok + pid_head * stride_k_head
    
    # 4. 前半部分和后半部分的列偏移 (Half-Half Layout)
    cols1 = tl.arange(0, BLOCK_SIZE)
    cols2 = cols1 + BLOCK_SIZE

    # 5. 加载 Q 的前后半部分
    q1 = tl.load(q_ptr + q_offset + cols1)
    q2 = tl.load(q_ptr + q_offset + cols2)
    
    # 加载 K 的前后半部分
    k1 = tl.load(k_ptr + k_offset + cols1)
    k2 = tl.load(k_ptr + k_offset + cols2)

    # 6. 加载预计算的 Cos 和 Sin 频率表
    cos_offset = pos * stride_cos_row
    cos = tl.load(cos_ptr + cos_offset + cols1)
    sin = tl.load(sin_ptr + cos_offset + cols1)

    # 7. 执行旋转计算
    q1_out = q1 * cos - q2 * sin
    q2_out = q2 * cos + q1 * sin
    
    k1_out = k1 * cos - k2 * sin
    k2_out = k2 * cos + k1 * sin

    # 8. 原位写回 (In-place) 覆盖原 Tensor 节省显存
    tl.store(q_ptr + q_offset + cols1, q1_out)
    tl.store(q_ptr + q_offset + cols2, q2_out)
    tl.store(k_ptr + k_offset + cols1, k1_out)
    tl.store(k_ptr + k_offset + cols2, k2_out)

def triton_apply_rope_inplace(q, k, cos, sin, positions):
    # q, k shape: [total_tokens, num_heads, head_dim]
    # cos, sin shape: [max_seq_len, head_dim // 2]
    # positions shape: [total_tokens]
    total_tokens = q.shape[0]
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    # RoPE 的半维度大小
    half_dim = head_dim // 2
    BLOCK_SIZE = triton.next_power_of_2(half_dim)
    # 2D Grid: (Tokens数量, Heads数量)
    grid = (total_tokens, num_heads)
    rope_inplace_kernel[grid](
        q, k, cos, sin, positions,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        cos.stride(0),
        head_dim=head_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return q, k
```