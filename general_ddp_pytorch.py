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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # Adjust input channels to 1 for grayscale MNIST images
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # Adjusted input size for fully connected layer
        self.fc2 = nn.Linear(128, 10)  # Output size adjusted for 10 classes in MNIST dataset

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(device, train_loader, model, loss_criterion, optimizer, profiler=None):
    train_loss = 0.0  # Initialize the cumulative training loss

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if profiler:
            profiler.step()

    avg_train_loss = train_loss / len(train_loader)
    print(f"Average Training Loss: {avg_train_loss:.4f}")


def validate(device, validation_loader, model, loss_criterion):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += loss_criterion(output, target).item()  # use the provided loss criterion
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(validation_loader.dataset)
    accuracy = 100. * correct / len(validation_loader.dataset)

    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')


def cleanup():
    dist.destroy_process_group()


def get_data(rank, world_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    validation_set = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)

    train_sampler = data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    validation_sampler = data.distributed.DistributedSampler(validation_set, num_replicas=world_size, rank=rank)

    train_loader = data.DataLoader(train_set, batch_size=32, sampler=train_sampler)
    validation_loader = data.DataLoader(validation_set, batch_size=64, sampler=validation_sampler)

    return train_loader, validation_loader


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


def main(rank, world_size):
    torch.manual_seed(0)
    train_loader, validation_loader = get_data(rank, world_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Rank {rank}: Using CUDA device {device}")

    model = Net().to(device)
    ddp_model = DistributedDataParallel(model)  # Wrap the model with DistributedDataParallel
    loss_criterion = nn.CrossEntropyLoss().to(device)  # Use CrossEntropyLoss for classification
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.05)

    for epoch in range(10):
        train(device, train_loader, ddp_model, loss_criterion, optimizer)
        validate(device, validation_loader, ddp_model, loss_criterion)
        scheduler.step()  # Place scheduler.step() here
    cleanup()


if __name__ == "__main__":
    try:
        torch.distributed.init_process_group(backend='gloo', init_method='env://')
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        main(rank, world_size)
    except Exception as e:
        print("An error occurred:", e)
        raise
