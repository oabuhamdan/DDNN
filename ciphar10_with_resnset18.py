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


def train(train_sampler, epoch, device, train_loader, model, loss_criterion, optimizer):
    train_sampler.set_epoch(epoch)
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
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


def validate(epoch, device, validation_loader, model, loss_criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in validation_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += loss_criterion(outputs, targets).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(validation_loader.dataset)
    accuracy = 100. * correct / len(validation_loader.dataset)
    print(f'\nValidation. Epoch: {epoch}, Loss: {test_loss:.4f}, Accuracy: ({accuracy:.2f}%)\n')


def cleanup():
    dist.destroy_process_group()


def get_data(rank, world_size):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_sampler = data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    train_loader = data.DataLoader(train_set, batch_size=64, sampler=train_sampler)

    validation_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    validation_loader = data.DataLoader(validation_set, batch_size=64)

    return train_sampler, train_loader, validation_loader


def on_trace_ready(prof):
    if rank == 0:
        print(prof.key_averages())

        profiler_output_dir = pathlib.Path("logs") / "profiler"
        profiler_output_dir.mkdir(exist_ok=True)

        output = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=32)
        print(f"Profile by total GPU time at step {prof.step_num}:\n{output}")
        output = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=32)
        print(f"Profile by total CPU time at step {prof.step_num}:\n{output}")

        prof.export_chrome_trace(
            str(trace_path := (profiler_output_dir / f"{prof.step_num}.chrome_trace.json.gz"))
        )


def main(args, rank, world_size):
    torch.manual_seed(0)
    train_sampler, train_loader, validation_loader = get_data(rank, world_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Rank {rank}: Using CUDA device {device}")

    model = models.resnet18(weights=None, num_classes=10).to(device)

    ddp_model = DistributedDataParallel(model)

    loss_criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.05)

    for epoch in range(args.epochs):
        train(train_sampler, epoch, device, train_loader, ddp_model, loss_criterion, optimizer)
        validate(epoch, device, validation_loader, ddp_model, loss_criterion)
        # scheduler.step()
    cleanup()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    try:
        torch.distributed.init_process_group(backend='gloo', init_method='env://')
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        main(args, rank, world_size)
    except Exception as e:
        print("An error occurred:", e)
        raise
