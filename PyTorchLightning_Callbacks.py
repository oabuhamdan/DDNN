import logging
import time

import torch
from lightning import Callback
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler


class ProfilerCallback(Callback):
    def __init__(self):
        self.start_time = None

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        print(f"Train_Started: {self.start_time}")
        torch.cuda.nvtx.range_push("Training")

    def on_train_end(self, trainer, pl_module):
        print(f"Train_Ended: {time.time()}")
        torch.cuda.nvtx.range_pop()
        # logging.info(f'{args.model},{args.strategy},{args.batch_size},{args.precision},{args.num_nodes},'
        #              f'{time.time() - self.start_time:.2f}')

    def on_before_backward(self, trainer, pl_module, loss):
        torch.cuda.nvtx.range_push("Backward")

    def on_after_backward(self, trainer, pl_module):
        torch.cuda.nvtx.range_pop()

    def on_validation_start(self, trainer, pl_module):
        torch.cuda.nvtx.range_push("Validation")

    def on_validation_end(self, trainer, pl_module):
        torch.cuda.nvtx.range_pop()

    def on_train_epoch_start(self, trainer, pl_module):
        torch.cuda.nvtx.range_push(f"Epoch_{trainer.current_epoch}")

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.nvtx.range_pop()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        torch.cuda.nvtx.range_push(f"Step_{batch_idx}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        torch.cuda.nvtx.range_pop()

    def on_before_setup_environment(self, trainer, pl_module):
        torch.cuda.nvtx.range_push(f"Environment_Setup")

    def on_after_setup_environment(self, trainer, pl_module):
        torch.cuda.nvtx.range_pop()

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        torch.cuda.nvtx.range_push(f"Optimizer_Step")

    def on_after_optimizer_step(self, trainer, pl_module, optimizer):
        torch.cuda.nvtx.range_pop()


class ThroughputCallback(Callback):
    def __init__(self):
        self.start_time = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        end_time = time.time()
        images, _ = batch
        local_batch_size = images.size(0)
        duration = end_time - self.start_time
        throughput = local_batch_size // duration
        pl_module.log('throughput', throughput, on_step=True, logger=True, sync_dist=True, reduce_fx="sum")
