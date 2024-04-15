import pandas as pd
import matplotlib.pyplot as plt


def get_utilization(path):
    data = pd.read_csv(path, delimiter='\s+')
    data = data[pd.to_numeric(data['gpu'], errors='coerce').notnull()]
    gpu_util = data['sm'].astype("int64")
    mem_util = data['mem'].astype("int64")
    return gpu_util, mem_util


gpu_util_1, mem_util_1 = get_utilization("logs/gpu_resnet50_cifar10_ddp_batch1024_nodes3/gpu_metrics_r1t01.csv")
gpu_util_2, mem_util_2 = get_utilization("logs/gpu_resnet50_cifar10_ddp_batch1024_nodes3/gpu_metrics_r1t02.csv")
gpu_util_3, mem_util_3 = get_utilization("logs/gpu_resnet50_cifar10_ddp_batch1024_nodes3/gpu_metrics_r1t03.csv")

fig, axes = plt.subplots(figsize=(7, 7), nrows=3, ncols=1)
axes[0].title.set_text("Resnet50 Cifar10 DDP 1024 Nodes")

gpu_util_1.plot(ax=axes[0], label="GPU Utilization")
mem_util_1.plot(ax=axes[0], label="Mem Utilization")
axes[0].text(0, 0, 'Node 1', fontsize=15, color="white", bbox={'facecolor': 'grey'})


gpu_util_2.plot(ax=axes[1], label="GPU Utilization")
mem_util_2.plot(ax=axes[1], label="Mem Utilization")
axes[0].text(0, 0, 'Node 2', fontsize=15, color="white", bbox={'facecolor': 'grey'})

gpu_util_3.plot(ax=axes[2], label="GPU Utilization")
mem_util_3.plot(ax=axes[2], label="Mem Utilization")
axes[0].text(0, 0, 'Node 3', fontsize=15, color="white", bbox={'facecolor': 'grey'})


plt.legend()
plt.tight_layout()
plt.savefig('figs/resnet50_cifar10_ddp_nodes_nvidia_smi_metrics.png')
