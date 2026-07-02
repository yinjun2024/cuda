import torch
import time

# 1. 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 2. 创建两个随机矩阵 (例如 1024x1024)
size = 4096
a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)

# 3. 预热 (可选，使 GPU 核函数加载完毕)
for _ in range(3):
    torch.mm(a, b)

# 4. 计时并执行矩阵乘法
torch.cuda.synchronize() if device.type == "cuda" else None
start = time.time()

c = torch.matmul(a, b)          # 矩阵乘法

torch.cuda.synchronize() if device.type == "cuda" else None
end = time.time()

# 5. 打印结果（只打印部分，避免刷屏）
print(f"结果矩阵形状: {c.shape}")
print(f"结果矩阵前5行5列:\n{c[:5, :5]}")

print(f"耗时: {(end - start) * 1000:.2f} ms")