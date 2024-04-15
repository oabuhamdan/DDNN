import pandas as pd
import matplotlib.pyplot as plt

def process_data(data):
    data = data.astype("float64")
    data = data / (1024 * 1024)
    return data


plt.figure(figsize=(12, 9))

plt.subplot(2, 1, 1)
dstat_node0 = pd.read_csv("logs/gpu_cifar10_resnet34_batch256/dstat_r1t00.csv", usecols=[2, 3, 4, 5])
dstat_node0_eth0_recv = process_data(dstat_node0.iloc[3:, 0])
dstat_node0_eth0_sent = process_data(dstat_node0.iloc[3:, 1])
dstat_node0_ib0_recv = process_data(dstat_node0.iloc[3:, 2])
dstat_node0_ib0_sent = process_data(dstat_node0.iloc[3:, 3])

dstat_node0 = pd.read_csv("logs/gpu_cifar10_resnet34_batch256/ibstat_r1t00.csv")
ibstat_node0_ib0_recv = process_data(dstat_node0.iloc[:, 1])
ibstat_node0_ib0_sent = process_data(dstat_node0.iloc[:, 2])

plt.plot(dstat_node0_ib0_recv, label="IB0 Dstat Recv", color="crimson", marker="^", markevery=25)
plt.plot(dstat_node0_ib0_sent, label="IB0 Dstat Sent", color="darkgreen", marker="v", markevery=25)

plt.plot(ibstat_node0_ib0_sent, label="IB0 Counter Sent", color="black", linestyle="--", dashes=(25, 1))
plt.plot(ibstat_node0_ib0_recv, label="IB0 Counter Recv", color="royalblue", markevery=25)
plt.ylabel("Throughput (MBps)", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
## ....................................... ##
plt.subplot(2, 1, 2)

dstat_node1 = pd.read_csv("logs/gpu_cifar10_resnet34_batch256/dstat_r1t02.csv", usecols=[2, 3, 4, 5])
dstat_node1_eth0_recv = process_data(dstat_node1.iloc[3:, 0])
dstat_node1_eth0_sent = process_data(dstat_node1.iloc[3:, 1])
dstat_node1_ib0_recv = process_data(dstat_node1.iloc[3:, 2])
dstat_node1_ib0_sent = process_data(dstat_node1.iloc[3:, 3])

dstat_node1 = pd.read_csv("logs/gpu_cifar10_resnet34_batch256/ibstat_r1t02.csv")
ibstat_node1_ib0_recv = process_data(dstat_node1.iloc[:, 1])
ibstat_node1_ib0_sent = process_data(dstat_node1.iloc[:, 2])

plt.plot(dstat_node1_ib0_recv, label="IB0 Dstat Recv", color="crimson", marker="^", markevery=25)
plt.plot(dstat_node1_ib0_sent, label="IB0 Dstat Sent", color="darkgreen", marker="v", markevery=25)

plt.plot(ibstat_node1_ib0_sent, label="IB0 Counter Sent", color="black", linestyle="--", dashes=(25, 1))
plt.plot(ibstat_node1_ib0_recv, label="IB0 Counter Recv", color="royalblue", markevery=25)
plt.ylabel("Throughput (MBps)", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.legend(loc='upper right', fontsize=12)
plt.show()
