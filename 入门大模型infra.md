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
