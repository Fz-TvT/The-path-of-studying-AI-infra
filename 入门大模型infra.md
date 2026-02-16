# Picotron
## dp naive 和 dp bucketing的区别
### Data Parallel Naive（朴素数据并行）
- 每个 GPU 完成整个反向传播（Backward Pass）后，逐个参数发起梯度同步
- 即：对每个参数张量 grad_i，单独执行一次 All-Reduce
