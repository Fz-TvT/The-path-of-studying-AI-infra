# Picotron
## dp naive 和 dp bucketing的区别
### Data Parallel Naive（朴素数据并行）
- 每个 GPU 完成整个反向传播（Backward Pass）后，逐个参数发起梯度同步
- 即：对每个参数张量 grad_i，单独执行一次 All-Reduce
# 伪代码示意（Naive DP）
```C++
            for param in model.parameters():
                compute_gradient(param)           # 反向传播计算梯度
                all_reduce(param.grad)           # 立即同步该参数的梯度 ← 阻塞式通信
```