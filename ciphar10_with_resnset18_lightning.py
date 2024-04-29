import argparse
import time

import torch
import torchmetrics
from lightning import Trainer, Callback
from lightning import seed_everything, LightningModule
from lightning.pytorch.loggers import CSVLogger
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms


class ProfilerCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        torch.cuda.nvtx.range_push("Training")

    def on_train_end(self, trainer, pl_module):
        torch.cuda.nvtx.range_pop()

    def on_validation_start(self, trainer, pl_module):
        torch.cuda.nvtx.range_push("Validation")

    def on_validation_end(self, trainer, pl_module):
        torch.cuda.nvtx.range_pop()

    def on_train_epoch_start(self, trainer, pl_module):
        torch.cuda.nvtx.range_push(f"Epoch {trainer.current_epoch}")

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.nvtx.range_pop()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        torch.cuda.nvtx.range_push(f"Train Epoch {trainer.current_epoch} batch {batch_idx}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        torch.cuda.nvtx.range_pop()


models = {
    "resnet18": models.resnet18,
    "resnet50": models.resnet50,
    "resnet152": models.resnet152,
}

precisions = {
    "fp16": torch.float16,
    "fp32": torch.float,
}


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
        if args.precision != "fp32":
            self.automatic_optimization = False
            self.scaler = GradScaler()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        start_time = time.time()

        images, labels = batch

        if self.automatic_optimization:
            out = self(images)
            loss = self.loss_function(out, labels)
        else:
            loss, out = self.manual_optimization(images, labels)

        self.train_acc(out, labels)
        samples_processed = images.size(0)
        time_elapsed = time.time() - start_time
        throughput = samples_processed // time_elapsed

        self.throughput.update(throughput)

        self.log('train_loss', loss, on_step=True, on_epoch=False, logger=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        return loss

    def manual_optimization(self, images, labels):
        optimizer = self.optimizers().optimizer
        optimizer.zero_grad()
        with autocast(dtype=precisions[args.precision]):
            out = self(images)
            loss = self.loss_function(out, labels)
        loss = self.scaler.scale(loss)
        self.manual_backward(loss)
        self.scaler.unscale_(optimizer)
        self.scaler.step(optimizer)
        self.scaler.update()
        return loss, out

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        return optimizer

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=1,
                                  persistent_workers=True)
        return train_loader

    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=1,
                                persistent_workers=True)
        return val_loader


def main():
    seed_everything(42)  # for reproducibility
    model = ResnetModel(loss_function=nn.CrossEntropyLoss(), num_classes=10)

    logger = CSVLogger(f"logs/{args.exp_name}/", name="csv_metrics")

    trainer = Trainer(
        max_epochs=args.epochs,
        strategy=args.strategy,
        accelerator="cuda",
        logger=logger,
        enable_progress_bar=True,
        num_nodes=args.num_nodes,
        log_every_n_steps=1,
        enable_model_summary=False,
        detect_anomaly=False,
        enable_checkpointing=False,
        callbacks=[ProfilerCallback()],
    )
    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--exp-name", type=str, default="test")
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--model", type=str, default="resnet18")
    args = parser.parse_args()
    main()
