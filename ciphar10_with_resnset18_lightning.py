import argparse

import torchmetrics
from lightning import Trainer
from lightning import seed_everything, LightningModule
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.plugins import FSDPPrecision, DeepSpeedPrecision, MixedPrecision
from lightning.pytorch.strategies import DeepSpeedStrategy, DDPStrategy, FSDPStrategy
from torch import nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms

from PyTorchLightning_Callbacks import *

logging.basicConfig(
    filename='report.csv',
    filemode='a',
    format="%(message)s",
    level=logging.INFO
)

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


model_dic = {
    "resnet50": models.resnet50,
    "resnet152": models.resnet152,
    "vgg19": models.vgg19_bn,
}


def get_modified_model(name, num_classes):
    model = model_dic[name](weights=None, num_classes=num_classes)
    if "resnet" in name:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    elif "vgg" in name:
        model.features[0] = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    return model


class Model(LightningModule):
    def __init__(self, loss_function, num_classes=10):
        super().__init__()
        self.model = get_modified_model(args.model, num_classes)
        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.valid_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.loss_function = loss_function

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        with torch.cuda.nvtx.range("Data_Loading"):
            images, labels = batch

        with torch.cuda.nvtx.range("Forward"):
            out = self(images)
            loss = self.loss_function(out, labels)

        self.train_acc(out, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=False, logger=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        out = self(images)
        self.valid_acc(out, labels)

        self.log('valid_acc', self.valid_acc, on_step=False, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        return optimizer

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                  pin_memory_device=str(self.device))
        return train_loader

    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR10(root="./data", train=False, download=False, transform=transform)
        val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                pin_memory_device=str(self.device))
        return val_loader


def main():
    seed_everything(42)  # for reproducibility

    model = Model(loss_function=nn.CrossEntropyLoss(), num_classes=10)

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
        callbacks=[ProfilerCallback(), ThroughputCallback()],
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
