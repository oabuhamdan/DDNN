import pandas as pd
import matplotlib.pyplot as plt


def get_data(path):
    df = pd.read_csv(path)
    received = df["rx_bytes"].astype("int64") / (1024 * 1024)
    sent = df["tx_bytes"].astype("int64") / (1024 * 1024)
    return received, sent


def filter_outliers(data):
    q = data.quantile(0.99)
    return data[data < q]


rec_ddp1, sent_ddp1 = get_data("logs/gpu_resnet50_cifar10_ddp_batch64_nodes3/ibstat_r1t02.csv")
rec_ddp2, sent_ddp2 = get_data("logs/gpu_resnet50_cifar10_ddp_batch64_nodes3/ibstat_r1t03.csv")
rec_ddp3, sent_ddp3 = get_data("logs/gpu_resnet50_cifar10_ddp_batch64_nodes3/ibstat_r1t06.csv")


fig, axes = plt.subplots(figsize=(7, 7), nrows=3, ncols=1)
axes[0].title.set_text("Resnet50 Cifar10 Nodes Batch1024")

rec_ddp1.plot(ax=axes[0], label="Received (MB/s)")
sent_ddp1.plot(ax=axes[0], label="Sent (MB/s)")
axes[0].text(0, 0, 'Node 1', fontsize=15, color="white", bbox={'facecolor': 'grey'})


rec_ddp2.plot(ax=axes[1], label="Received (MB/s)")
sent_ddp2.plot(ax=axes[1], label="Sent (MB/s)")
axes[1].text(0, 0, 'Node 2', fontsize=15, color="white", bbox={'facecolor': 'grey'})


rec_ddp3.plot(ax=axes[2], label="Received (MB/s)")
sent_ddp3.plot(ax=axes[2], label="Sent (MB/s)")
axes[2].text(0, 0, 'Node 3', fontsize=15, color="white", bbox={'facecolor': 'grey'})


plt.legend()
plt.tight_layout()
plt.savefig('figs/resnet50_cifar10_ddp_64_nodes_ibstat.png')
