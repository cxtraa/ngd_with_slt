import torch
import time

def get_gpu_memory_usage(device_id):
    """
    Returns the current GPU memory usage by tensors in bytes for a given device
    """
    return torch.cuda.memory_allocated(device_id)

def print_gpu_utilization():
    """
    Prints the GPU utilization and memory usage
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory Allocated: {get_gpu_memory_usage(0) / (1024 ** 3):.2f} GB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f} GB")
    else:
        print("CUDA is not available. Running on CPU.")

def heavy_compute():
    """
    Perform a heavy compute task
    """
    print("Starting heavy computation...")

    # Ensure that PyTorch is using the GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Large matrix multiplication
    n = 120000 # Size of the matrix (n x n)
    a = torch.rand(n, n, device=device)
    b = torch.rand(n, n, device=device)

    start_time = time.time()
    c = torch.matmul(a, b)
    end_time = time.time()

    print(f"Completed in {end_time - start_time:.2f} seconds.")
    return c

print_gpu_utilization()
result = heavy_compute()
print_gpu_utilization()
