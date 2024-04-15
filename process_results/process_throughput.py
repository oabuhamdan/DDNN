import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def get_data(path):
    data = pd.read_csv(path)
    throughput = data['Value'].astype("int64")
    return throughput


resnet18_ddp_64 = get_data("logs/gpu_resnet50_cifar10_ddp_batch1024_nodes3/overall_throughput.csv")
resnet18_ddp_512 = get_data("logs/gpu_resnet50_cifar10_fsdp_batch1024_nodes3/overall_throughput.csv")
resnet18_ddp_1024 = get_data("logs/gpu_resnet50_cifar10_deepspeed_stage_2_batch1024_nodes3/overall_throughput.csv")

# resnet18_deepspeed_1024 = get_data("logs/gpu_resnet18_cifar10_deepspeed_stage_2_batch1024_nodes3/overall_throughput.csv")
# resnet18_fsdp_1024 = get_data("logs/gpu_resnet18_cifar10_fsdp_batch1024_nodes3/overall_throughput.csv")

plt.figure(figsize=(8, 6))
plt.plot(resnet18_ddp_64, label="Resnet50 - DDP - Batch 1024")
plt.plot(resnet18_ddp_512, label="Resnet50 - FSDP - Batch 1024")
plt.plot(resnet18_ddp_1024, label="Resnet50 - DeepSpeed2 - Batch 1024")
# plt.plot(resnet18_fsdp_1024, label="Resnet18 - FSDP - Batch 1024")
# plt.plot(resnet18_deepspeed_1024, label="Resnet18 - DeepSpeed2 - Batch 1024")

plt.xlabel("Time (s)")
plt.ylabel("Throughput (Images/s)")
plt.legend()
plt.tight_layout()
plt.savefig('figs/resnet50_cifar10_batch1024_strategies_throughput.png')
