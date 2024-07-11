import pathlib

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from torchvision import models
import argparse
import horovod.torch as hvd


def train(train_sampler, epoch, train_loader, model, loss_criterion, optimizer, profiler):
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        profiler.step()
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_criterion(output, targets)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            current_samples = batch_idx * len(inputs)
            total_samples = len(train_sampler)
            progress = 100.0 * current_samples / total_samples
            print(f'Epoch: {epoch} [{current_samples}/{total_samples} ({progress:.0f}%)]\tLoss: {loss.item():.6f}')


def validate(epoch, validation_loader, model, loss_criterion):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in validation_loader:
            outputs = model(inputs)
            test_loss += loss_criterion(outputs, targets).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(validation_loader.dataset)
    accuracy = 100. * correct / len(validation_loader.dataset)
    print(f'\nValidation. Epoch: {epoch}, Loss: {test_loss:.4f}, Accuracy: ({accuracy:.2f}%)\n')


def cleanup():
    dist.destroy_process_group()


def get_data():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_sampler = data.distributed.DistributedSampler(train_set, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = data.DataLoader(train_set, batch_size=512, sampler=train_sampler)

    validation_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    validation_loader = data.DataLoader(validation_set, batch_size=512)

    return train_sampler, train_loader, validation_loader


def main(args):
    torch.manual_seed(0)
    train_sampler, train_loader, validation_loader = get_data()

    print(f"Rank {hvd.rank()}")

    model = models.resnet18(weights=None, num_classes=10)
    model.cuda()

    model = DistributedDataParallel(model)

    loss_criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/example'),
    ) as p:
        for epoch in range(args.epochs):
            train(train_sampler, epoch, train_loader, model, loss_criterion, optimizer, p)
            validate(epoch, validation_loader, model, loss_criterion)
        cleanup()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=1)
    return parser.parse_args()


hvd.init()
torch.cuda.set_device(hvd.local_rank())
args = get_args()
main(args)

# if __name__ == "__main__":
#     args = get_args()
#     try:
#         torch.distributed.init_process_group(backend='nccl', init_method='env://')
#         rank = torch.distributed.get_rank()
#         world_size = torch.distributed.get_world_size()
#         main(args, rank, world_size)
#     except Exception as e:
#         print("An error occurred:", e)
#         raise
