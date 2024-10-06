import argparse
import csv
import json
import os
import time
import logging
from collections import OrderedDict
from dataclasses import asdict, dataclass
from importlib.metadata import version
from itertools import zip_longest
from typing import Any, Callable, Dict, List, Optional, Union
from importlib.metadata import version

import torch

# Set up logging configuration
logging.basicConfig(level=logging.INFO)

LIGER_KERNEL_VERSION = version("liger-kernel")
QUANTILES = [0.5, 0.2, 0.8]

@dataclass
class SingleBenchmarkRunInput:
    x: Union[int, float]
    kernel_provider: str
    kernel_operation_mode: Optional[str] = ""
    extra_benchmark_config: Optional[Dict[str, Any]] = None

@dataclass
class SingleBenchmarkRunOutput:
    # 20th percentile
    y_20: float
    # 50th percentile (median)
    y_50: float
    # 80th percentile
    y_80: float

@dataclass
class BenchmarkData:
    """
    BenchmarkData is a dataclass to store the benchmark data for a completed benchmark
    run on all x-values for a given kernel/kernel operation mode/metric/extra_benchmark_config
    """
    kernel_name: str
    kernel_provider: str
    metric_name: str
    metric_unit: str
    gpu_name: str
    x_name: str
    x_label: str
    x_values: List[float]
    y_values_50: List[float]
    y_values_20: List[float]
    y_values_80: List[float]
    timestamp: str
    kernel_operation_mode: Optional[str] = None
    extra_benchmark_config_str: Optional[str] = None
    liger_version: str = LIGER_KERNEL_VERSION

@dataclass
class BenchmarkDataCSVRow:
    kernel_name: str
    kernel_provider: str
    kernel_operation_mode: Optional[str]
    metric_name: str
    metric_unit: str
    x_name: str
    x_label: str
    x_value: float
    y_value_50: float
    y_value_20: float
    y_value_80: float
    extra_benchmark_config_str: Optional[str]
    gpu_name: str
    timestamp: str
    liger_version: str

def _test_memory(
    func: Callable,
    _iter: int = 10,
    quantiles: Optional[List[float]] = None,
    return_mode="mean",
) -> float:
    """
    Tests the memory usage of a given function over a specified number of iterations.
    
    Args:
        func (Callable): The function to be tested.
        _iter (int): Number of iterations to test memory.
        quantiles (Optional[List[float]]): List of quantiles to compute.
        return_mode (str): Specifies the aggregation method for memory (min, max, mean, median).

    Returns:
        float or List[float]: Memory usage statistics.
    """
    assert return_mode in ["min", "max", "mean", "median"], "Invalid return mode"
    
    total_mem = []

    for _ in range(_iter):
        torch.cuda.memory.reset_peak_memory_stats()
        func()
        mem = torch.cuda.max_memory_allocated() / 2**20  # Convert to MB
        total_mem.append(mem)

    total_mem = torch.tensor(total_mem, dtype=torch.float)
    
    if quantiles:
        quantiles_data = torch.quantile(total_mem, torch.tensor(quantiles, dtype=torch.float)).tolist()
        return quantiles_data[0] if len(quantiles_data) == 1 else quantiles_data
    
    return getattr(torch, return_mode)(total_mem).item()

def get_current_file_directory() -> str:
    """Returns the directory path of the current Python file."""
    current_file_path = os.path.abspath(__file__)
    return os.path.dirname(current_file_path)

def sleep(seconds):
    """Decorator to pause execution for a specified number of seconds."""
    def decorator(function):
        def wrapper(*args, **kwargs):
            time.sleep(seconds)
            return function(*args, **kwargs)
        return wrapper
    return decorator

def _print_benchmarking_banner(metric_name: str, kernel_name: str):
    """Prints a banner for the benchmarking process."""
    print("**************************************")
    print(f"     BENCHMARKING {metric_name.upper()} for {kernel_name.upper()}")
    print("**************************************")

def get_formatted_time() -> str:
    """Returns the current formatted time as a string."""
    return time.strftime("%Y-%m-%d %H:%M:%S")

def get_gpu_name() -> str:
    """Returns the current GPU name, formatted to serve as a directory name."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        return gpu_name
    raise Exception("Benchmarks can only be run on GPU. Please ensure your environment has a GPU available.")

def update_benchmark_data_csv(
    benchmark_data_list: List[BenchmarkData],
    filename: str = "all_benchmark_data.csv",
    overwrite: bool = True,
):
    """
    Update the CSV file with the new benchmark data. If the file does not exist, create it.
    
    Args:
        benchmark_data_list (List[BenchmarkData]): List of benchmark data to update.
        filename (str): Name of the CSV file to update.
        overwrite (bool): Whether to overwrite existing benchmark data entries.
    """
    logging.info(f"Updating benchmark data in {filename}")

    if not benchmark_data_list:
        logging.warning("No benchmark data to update.")
        return

    def create_unique_key(row):
        return (
            row["kernel_name"],
            row["kernel_provider"],
            row["kernel_operation_mode"] if row["kernel_operation_mode"] else "",
            row["metric_name"],
            row["x_name"],
            str(row["x_value"]),
            row["extra_benchmark_config_str"] if row["extra_benchmark_config_str"] else "",
            row["gpu_name"],
        )

    fieldnames = BenchmarkDataCSVRow.__annotations__.keys()

    filename_abs_path = os.path.join(get_current_file_directory(), "../data", filename)
    file_exists = os.path.isfile(filename_abs_path)

    existing_data = []
    if file_exists:
        with open(filename_abs_path, mode="r") as file:
            reader = csv.DictReader(file)
            existing_data = [row for row in reader]

    existing_data_dict = OrderedDict((create_unique_key(row), row) for row in existing_data)

    for benchmark_data in benchmark_data_list:
        benchmark_data_dict = asdict(benchmark_data)
        x_values = benchmark_data_dict.pop("x_values")
        y_values_50 = benchmark_data_dict.pop("y_values_50")
        y_values_20 = benchmark_data_dict.pop("y_values_20")
        y_values_80 = benchmark_data_dict.pop("y_values_80")

        for x_value, y_value_50, y_value_20, y_value_80 in zip_longest(x_values, y_values_50, y_values_20, y_values_80):
            row = BenchmarkDataCSVRow(
                x_value=x_value,
                y_value_50=y_value_50,
                y_value_20=y_value_20,
                y_value_80=y_value_80,
                **benchmark_data_dict,
            )
            row_dict = asdict(row)
            row_key = create_unique_key(row_dict)

            if row_key in existing_data_dict:
                if overwrite:
                    existing_data_dict[row_key] = row_dict
            else:
                existing_data_dict[row_key] = row_dict

    with open(filename_abs_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing_data_dict.values():
            writer.writerow(row)

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.dtype):
            return str(obj)
        return super().default(obj)

def print_benchmark_data(benchmark_data_list: List[BenchmarkData]) -> None:
    """Prints the benchmark data in a formatted JSON style."""
    print("********** Benchmark Data **********")
    formatted_list = [obj.__dict__ for obj in benchmark_data_list]
    print(json.dumps(formatted_list, indent=2))

def run_benchmarks(
    bench_test_fn: Callable,
    kernel_name: str,
    metric_name: str,
    metric_unit: str,
    x_name: str,
    x_label: str,
    x_values: List[Union[float, int]],
    kernel_providers: List[str],
    kernel_operation_modes: Optional[List[str]] = None,
    extra_benchmark_configs: Optional[List[Dict[str, Any]]] = None,
    overwrite: bool = False,
):
    """
    Run benchmarks given a bench_test_fn that takes in a SingleBenchmarkRunInput as input and
    saves data to the CSV file.

    Args:
        bench_test_fn (Callable): The benchmark test function to run.
        kernel_name (str): The name of the kernel being benchmarked.
        metric_name (str): The name of the metric being benchmarked.
        metric_unit (str): The unit of the metric being benchmarked.
        x_name (str): The name of the x-axis.
        x_label (str): The label of the x-axis.
        x_values (List[Union[float, int]]): The list of x-values to run the benchmark on.
        kernel_providers (List[str]): List of kernel providers.
        kernel_operation_modes (Optional[List[str]]): List of kernel operation modes.
        extra_benchmark_configs (Optional[List[Dict[str, Any]]]): List of extra benchmark configurations.
        overwrite (bool): Whether to overwrite existing benchmark data entries.
    """
    if kernel_operation_modes is None:
        kernel_operation_modes = [None]

    if extra_benchmark_configs is None:
        extra_benchmark_configs = [None]

    benchmark_data_list = []

    for kernel_provider in kernel_providers:
        for kernel_operation_mode in kernel_operation_modes:
            for extra_benchmark_config in extra_benchmark_configs:
                _print_benchmarking_banner(metric_name, kernel_name)

                y_values_50, y_values_20, y_values_80 = [], [], []

                for x in x_values:
                    input_data = SingleBenchmarkRunInput(x=x, kernel_provider=kernel_provider,
                                                          kernel_operation_mode=kernel_operation_mode,
                                                          extra_benchmark_config=extra_benchmark_config)
                    outputs = bench_test_fn(input_data)

                    y_values_50.append(outputs.y_50)
                    y_values_20.append(outputs.y_20)
                    y_values_80.append(outputs.y_80)

                benchmark_data = BenchmarkData(
                    kernel_name=kernel_name,
                    kernel_provider=kernel_provider,
                    metric_name=metric_name,
                    metric_unit=metric_unit,
                    gpu_name=get_gpu_name(),
                    x_name=x_name,
                    x_label=x_label,
                    x_values=x_values,
                    y_values_50=y_values_50,
                    y_values_20=y_values_20,
                    y_values_80=y_values_80,
                    timestamp=get_formatted_time(),
                    kernel_operation_mode=kernel_operation_mode,
                    extra_benchmark_config_str=json.dumps(extra_benchmark_config, cls=CustomEncoder) if extra_benchmark_config else None
                )

                benchmark_data_list.append(benchmark_data)

    print_benchmark_data(benchmark_data_list)
    update_benchmark_data_csv(benchmark_data_list, overwrite=overwrite)

def main():
    parser = argparse.ArgumentParser(description="Benchmarking script")
    # Add your command-line arguments here
    args = parser.parse_args()
    # Call the run_benchmarks function here with the necessary parameters

if __name__ == "__main__":
    main()
