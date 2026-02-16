# Picotron
## dp naive 和 dp bucketing的区别
### Data Parallel Naive（朴素数据并行）
- 每个 GPU 完成整个反向传播（Backward Pass）后，逐个参数发起梯度同步
- 即：对每个参数张量 grad_i，单独执行一次 All-Reduce
### DP with Bucketing（分桶优化的 DP）
- 将多个小梯度张量合并成一个“桶”（Bucket）（如 25MB）
- 反向传播过程中，一旦桶满，立即异步启动 All-Reduce
- 实现 “边计算边通信”（计算-通信重叠）