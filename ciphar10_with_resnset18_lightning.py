import argparse
import time

import torch
import torchmetrics
from lightning import Trainer, Callback
from lightning import seed_everything, LightningModule
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.plugins import FSDPPrecision, DeepSpeedPrecision, MixedPrecision
from torch import nn
from torch.cuda.amp import GradScaler
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
from lightning.pytorch.strategies import DeepSpeedStrategy, DDPStrategy, FSDPStrategy
from deepspeed.ops.adam import DeepSpeedCPUAdam
import logging

logging.basicConfig(
    filename='report.csv',
    filemode='a',
    format="%(message)s",
    level=logging.INFO
)

models = {
    "resnet18": models.resnet18,
    "resnet50": models.resnet50,
    "resnet152": models.resnet152,
}

deepspeed_config = dict(process_group_backend="nccl", allgather_bucket_size=5e8, reduce_bucket_size=5e8)

strategies = {
    "ddp": DDPStrategy(process_group_backend="nccl"),
    "fsdp": FSDPStrategy(process_group_backend="nccl"),
    "deepspeed": DeepSpeedStrategy(**deepspeed_config, stage=3),
    "deepspeed_offload": DeepSpeedStrategy(**deepspeed_config, stage=3, offload_optimizer=True, offload_parameters=True)
}


def get_plugins(args):
    if "16" in args.precision:
        if "fsdp" in args.strategy:
            return [FSDPPrecision(precision="16-true", scaler=ShardedGradScaler())]
        if "deepspeed" in args.strategy:
            return [DeepSpeedPrecision(precision="16-true")]
        if "ddp" in args.strategy:
            return [MixedPrecision(precision="16-mixed", device="cuda", scaler=GradScaler())]
    return []


class ProfilerCallback(Callback):
    def __init__(self, profiler):
        self.profiler = profiler
        self.start_time = None

    def on_train_start(self, trainer, pl_module):
        self.profiler.start()
        self.start_time = time.time()
        print(f"Train_Started: {self.start_time}")

    def on_train_end(self, trainer, pl_module):
        print(f"Train_Ended: {time.time()}")
        logging.info(
            f'{args.model},{args.strategy},{args.batch_size},{args.precision},{args.num_nodes},'
            f'{time.time() - self.start_time:.2f}')
        self.profiler.stop()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.profiler.step()


class ResnetModel(LightningModule):
    def __init__(self, loss_function, num_classes=10):
        super().__init__()
        model = models[args.model](weights=None, num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model = model
        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.valid_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.throughput = torchmetrics.SumMetric()
        self.loss_function = loss_function

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        start_time = time.time()

        images, labels = batch
        out = self(images)
        loss = self.loss_function(out, labels)

        self.train_acc(out, labels)
        samples_processed = images.size(0)
        time_elapsed = time.time() - start_time
        throughput = samples_processed // time_elapsed
        self.throughput.update(throughput)

        self.log('train_loss', loss, on_step=True, on_epoch=False, logger=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.log("throughput", self.throughput.compute().item())
        self.throughput.reset()

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        out = self(images)
        loss = self.loss_function(out, labels)
        self.valid_acc(out, labels)

        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('valid_acc', self.valid_acc, on_step=False, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        if "offload" in args.strategy:
            optimizer = DeepSpeedCPUAdam(self.parameters(), lr=0.001, weight_decay=1e-4, adamw_mode=True)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), weight_decay=1e-4, lr=0.001)
        return optimizer

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        return train_loader

    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR10(root="./data", train=False, download=False, transform=transform)
        val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        return val_loader


def main():
    seed_everything(42)  # for reproducibility

    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"logs/{args.exp_name}/"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )

    model = ResnetModel(loss_function=nn.CrossEntropyLoss(), num_classes=10)

    logger = CSVLogger(f"logs/{args.exp_name}/", name="csv_metrics")

    trainer = Trainer(
        max_epochs=args.epochs,
        strategy=strategies[args.strategy],
        accelerator="cuda",
        logger=logger,
        enable_progress_bar=True,
        num_nodes=args.num_nodes,
        log_every_n_steps=1,
        enable_model_summary=True,
        detect_anomaly=False,
        enable_checkpointing=False,
        callbacks=[ProfilerCallback(profiler=prof)],
        plugins=get_plugins(args)
    )

    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--exp-name", type=str, default="test")
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--model", type=str, default="resnet18")
    args = parser.parse_args()
    main()
