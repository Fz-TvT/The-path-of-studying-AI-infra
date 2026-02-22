# Picotron
## 数据并行
数据并行分为DP（Data Parallel）、DDP （Distributed Data Parallel）和 FSDP（Fully Sharded Data Parallel）

- DP:一个老板（主进程）管多个员工（GPU），所有员工的工作内容、参数都一样，老板统一分发任务、收集结果、更新参数
- DDP:多机多卡 / 单机多卡场景下，每个 GPU 对应一个独立进程，每个进程都持有完整的模型参数。通过 NCCL 通信库实现梯度的高效同步（Ring AllReduce 算法），每个进程独立更新参数（因为梯度同步后参数一致）
- 基于 DDP 改进，解决大模型显存不足问题。将模型参数、梯度、优化器状态都 “分片”（Shard）到不同的 GPU / 节点上，每个 GPU 只持有模型的一部分数据，训练时按需加载 / 卸载分片，通过通信同步分片数据。关键特性：支持 “ZeRO 优化”（零冗余优化器），分为 ZeRO-1（分片优化器状态）、ZeRO-2（分片梯度 + 优化器状态）、ZeRO-3（分片参数 + 梯度 + 优化器状态），FSDP 默认接近 ZeRO-3
## Pytorch DDP 中dp naive 和 dp bucketing的区别
### Data Parallel Naive（朴素数据并行）
**特点**
- 每个 GPU 完成整个反向传播（Backward Pass）后，逐个参数发起梯度同步
- 即：对每个参数张量 $grad_i$，单独执行一次 All-Reduce

PyTorch 自动求导引擎能够接纳自定义反向钩子。DDP 通过注册自动求导钩子，在每次反向传播结束后触发计算流程。钩子触发时，会全面扫描所有本地模型参数，从各个参数里获取梯度张量。随后，对每个参数运用 AllReduce 集体通信（对所有进程的该参数的梯度进行求和或者取平均），计算所有进程中每个参数的平均梯度，并将结果回写到梯度张量之中。朴素解决方案虽可行但存在以下不足：

- 对于大型模型中大量的小参数，AllReduce 操作在传输这些小梯度张量时效率较低。通信启动和同步的开销相对于小张量的实际数据量占比过大，导致传输过程浪费了大量时间。小张量通信效率低下会累积延迟，成为分布式训练的性能瓶颈。
- 梯度计算和梯度同步被分为两个独立阶段，二者串行进行。梯度计算完成后才启动通信，而通信完成后才继续下一步计算。这种串行设计导致计算设备在通信期间处于空闲状态，无法充分利用资源，降低了训练效率。
``` Python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import argparse
import os

def setup():
    """初始化分布式环境"""
    dist.init_process_group(backend="nccl")  # GPU 推荐 nccl
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        return self.net(x)

def main():
    setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # 创建模型、优化器
    model = SimpleModel().to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    # 模拟数据集
    dataset = TensorDataset(torch.randn(1000, 100), torch.randint(0, 10, (1000,)))
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=local_rank
    )
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 训练循环
    model.train()
    for epoch in range(3):
        sampler.set_epoch(epoch)  # 确保每个 epoch 数据打乱不同
        for x, y in dataloader:
            x, y = x.to(local_rank), y.to(local_rank)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
        if local_rank == 0:  # 只让 rank 0 打印
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    cleanup()

if __name__ == "__main__":
    main()
```
### DP with Bucketing（分桶优化的 DP）
**特点**
- 将多个小梯度张量合并成一个“桶”（Bucket）（如 25MB）
- 反向传播过程中，一旦桶满，立即异步启动 All-Reduce
- 实现 “边计算边通信”（计算-通信重叠）

# nano-vllm
## Prefill /Decode/KV Cache
- **KV Cache**:
没有KV Cache时如何计算Transformer:
    <details>
    <summary>Python代码</summary>

    Step1
    ```Python
    # 输入序列
    X₁ = [P₁, P₂, P₃]  # 3个token
    # 计算 QKV（3个token都要算）
    Q₁ = X₁ @ Wq  # [3, d]
    K₁ = X₁ @ Wk  # [3, d]  ← 计算了P₁,P₂,P₃的K
    V₁ = X₁ @ Wv  # [3, d]  ← 计算了P₁,P₂,P₃的V
    # Attention
    attn = softmax(Q₁ @ K₁.T / √d)  # [3, 3]
    output = attn @ V₁
    # 取最后一个token作为输出
    O₁ = output[-1]
    ```
    Step2:
    ```Python
    # 输入序列（变长了）
    X₂ = [P₁, P₂, P₃, O₁]  # 4个token
    # 计算 QKV（4个token都要算，包括重复的P!）
    Q₂ = X₂ @ Wq  # [4, d]
    K₂ = X₂ @ Wk  # [4, d]  ← 又计算了P₁,P₂,P₃的K！浪费！
    V₂ = X₂ @ Wv  # [4, d]  ← 又计算了P₁,P₂,P₃的V！浪费！
    # Attention
    attn = softmax(Q₂ @ K₂.T / √d)  # [4, 4]
    output = attn @ V₂
    O₂ = output[-1]
    ```
    </details>

    有KV Cache时如何计算Transformer:
    <details>
    <summary>Python代码</summary>

    Prefill 阶段（构建缓存）
    ```Python
    import math
        # 输入: [P₁, P₂, P₃, ..., Pₙ]  n个prompt token
        # 对每一层 Transformer:
        for layer in layers:
            # 并行计算所有token的QKV
            Q = X @ Wq  # [n, d]
            K = X @ Wk  # [n, d]  ← 缓存
            V = X @ Wv  # [n, d]  ← 缓存
            # 存储到KV Cache
            kv_cache[layer] = {
                'K': K,  # [n, d]
                'V': V   # [n, d]
            }
            # 计算Attention输出
            output = softmax(Q @ K.T / math.sqrt(d)) @ V
    ```
     Decode 阶段（复用缓存）
    ```Python
    import math
        # 生成第1个token O₁:
    new_token = O₁
    for layer in layers:
        # 只计算新token的QKV
        q = new_token @ Wq  # [1, d]
        k = new_token @ Wk  # [1, d]
        v = new_token @ Wv  # [1, d]
        
        # 追加到缓存
        kv_cache[layer]['K'] = concat(kv_cache[layer]['K'], k)  # [n+1, d]
        kv_cache[layer]['V'] = concat(kv_cache[layer]['V'], v)  # [n+1, d]
        
        # Attention: Q只查询，KV用缓存
        output = softmax(q @ K_cache.T / math.sqrt(d)) @ V_cache
    ```
- 为什么Q不需要缓存:
    Q是当前token的查询向量。K是所有token的键向量。V是所有token的值向量。Q用完即弃，历史Q不会再被使用，缓存没有意义。

- 总推理时间 = Prefill 时间 + Decode 时间 × 生成 token 数
- 短 Prompt + 长回复：Decode 主导耗时
- 长 Prompt + 短回复：Prefill 主导耗时
┌─────────────────────────────────────────┐
│           Prefill 阶段                   │
│  输入: ["请", "解释", "Transformer"]     │
│        ↓ 并行计算                        │
│  输出: KV Cache + 第一个生成 token       │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│           Decode 阶段                    │
│  输入: token₁ → token₂ → token₃ → ...   │
│        ↓ 串行迭代                        │
│  输出: 完整回复内容                      │
└─────────────────────────────────────────┘
