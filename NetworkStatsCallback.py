import logging
import time

import psutil
from lightning import Callback
from typing_extensions import override


class NetworkStatsCallback(Callback):
    def __init__(self, exp_name):
        self.node_rank = 0
        self.local_rank = 0
        self.start_time = 0
        self.step_start_time = 0
        self.net_counters = None
        self.net_stats_logger = logging.getLogger("net_stat_logger")
        self.net_stats_logger.setLevel(logging.INFO)
        self.net_stats_logger.addHandler(
            logging.FileHandler(filename=f'logs/{exp_name}/net_stats.csv', encoding='utf-8'))

    @override
    def setup(self, trainer, pl_module, stage):
        self.start_time = time.time()
        self.node_rank = trainer.node_rank
        self.local_rank = trainer.local_rank
        self.net_stats_logger.info("wall_clock,relative_time,step_duration,delta_recv,delta_sent")

    @override
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # TODO: Fix for multiple devices in one node
        self.step_start_time = time.time()
        self.net_counters = psutil.net_io_counters(nowrap=True)

    @override
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # TODO: Fix for multiple devices in one node
        abs_time = time.time()
        relative_time = round(abs_time - self.start_time, 2)
        step_dur = round(abs_time - self.step_start_time, 2)
        delta_recv, delta_sent = self.delta_net_stats()
        if self.local_rank == 0:
            msg = self.get_data_formatted_to_csv(relative_time, step_dur, delta_recv, delta_sent)
            self.net_stats_logger.info(msg)

    def delta_net_stats(self):
        net_counters = psutil.net_io_counters(nowrap=True)
        delta_recv = net_counters.bytes_recv - self.net_counters.bytes_recv
        delta_sent = net_counters.bytes_sent - self.net_counters.bytes_sent
        return delta_recv, delta_sent

    @staticmethod
    def get_data_formatted_to_csv(rel_time, step_dur, delta_recv, delta_sent):
        return f"{time.strftime('%H:%M:%S')},{rel_time},{step_dur},{delta_recv},{delta_sent}"
