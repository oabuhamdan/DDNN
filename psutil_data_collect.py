import argparse
import json
import os
import time
import asyncio
import psutil


def bytes_to_megabytes(value):
    return round(value / (1024 * 1024), 3)


async def collect_data(start_time, prev_io_counters, prev_swap_counters, prev_net_counters):
    elapsed_time = time.time() - start_time

    cpu_percent = psutil.cpu_percent(interval=args.interval)
    mem_usage = psutil.virtual_memory().percent
    swap_usage = psutil.swap_memory().percent

    io_counters = psutil.disk_io_counters()
    net_counters = psutil.net_io_counters()
    swap_counters = psutil.swap_memory()

    # IO DATA
    delta_read_bytes = io_counters.read_bytes - prev_io_counters.read_bytes
    delta_write_bytes = io_counters.write_bytes - prev_io_counters.write_bytes

    # Compute delta swap memory counters
    delta_sin = swap_counters.sin - prev_swap_counters.sin
    delta_sout = swap_counters.sout - prev_swap_counters.sout

    # Network Data
    delta_bytes_sent = net_counters.bytes_sent - prev_net_counters.bytes_sent
    delta_bytes_received = net_counters.bytes_recv - prev_net_counters.bytes_recv
    delta_errin = net_counters.errin - prev_net_counters.errin
    delta_errout = net_counters.errout - prev_net_counters.errout
    delta_dropin = net_counters.dropin - prev_net_counters.dropin
    delta_dropout = net_counters.dropout - prev_net_counters.dropout

    data = {
        'Timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        'Elapsed Time': round(elapsed_time),
        'CPU Usage': cpu_percent,
        'Memory Usage': mem_usage,
        'Swap Usage': swap_usage,
        'MEM Pages in': bytes_to_megabytes(delta_sin),
        'MEM Pages out': bytes_to_megabytes(delta_sout),
        'IO Delta Read Bytes': bytes_to_megabytes(delta_read_bytes),
        'IO Delta Write Bytes': bytes_to_megabytes(delta_write_bytes),
        'NET Delta Bytes Sent': bytes_to_megabytes(delta_bytes_sent),
        'NET Delta Bytes Received': bytes_to_megabytes(delta_bytes_received),
        'NET Delta Errin': delta_errin,
        'NET Delta Errout': delta_errout,
        'NET Delta Dropin': delta_dropin,
        'NET Delta Dropout': delta_dropout
    }
    return data, io_counters, net_counters, swap_counters


async def write_to_json(data):
    os.makedirs("logs/psutil_logs", exist_ok=True)
    with open(os.path.join("logs/psutil_logs", args.output), 'a') as file:
        json.dump(data, file)


async def main():
    all_data = []
    start_time = time.time()
    io_counters = psutil.disk_io_counters()
    net_counters = psutil.net_io_counters()
    swap_counters = psutil.swap_memory()
    try:
        while True:
            data, io_counters, net_counters, swap_counters = await collect_data(start_time, io_counters,
                                                                                swap_counters, net_counters)
            all_data.append(data)
            await asyncio.sleep(args.interval)
    except KeyboardInterrupt:
        print("Data collection interrupted by user.")
    finally:
        await write_to_json(all_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect system resource data and write to JSON file')
    parser.add_argument('--output', type=str)
    parser.add_argument('--interface', type=str)
    parser.add_argument('--interval', type=int, default=5)
    args = parser.parse_args()
    asyncio.run(main())
